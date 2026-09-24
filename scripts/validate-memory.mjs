// Developer-only comparison against fixtures from validate-memory.py.
import { chromium } from '@playwright/test';
import { readFile, writeFile } from 'node:fs/promises';
import ort from 'onnxruntime-node';
const root = 'artifacts/memory-validation';
const fixtures = JSON.parse(await readFile(`${root}/fixtures.json`, 'utf8'));
const manifest = JSON.parse(await readFile('static/models/sam2-large/manifest.json', 'utf8'));
const tasks = [
    ...[true, false].map(prompt => ({ part: 'memoryEncoder', name: prompt ? 'prompt-memory' : 'tracked-memory',
        inputs: { features: 'features', mask: 'mask', is_prompt: { dims: [1], data: [prompt ? 1 : 0] } },
        outputs: { memory: prompt ? 'memory-prompt' : 'memory-tracked' } })),
    ...[1, 3, 7].map(n => ({ part: 'memoryFusion', name: `fusion-${n}`, inputs: { features: 'features', memories: `memories-${n}`, memory_indices: `indices-${n}`, pointers: `pointers-${n}` }, outputs: { fused_features: `fused-${n}` } })),
    ...[1, 3, 7].map(n => ({ part: 'decoder', name: `tracking-decoder-${n}`, inputs: { features: `fused-${n}`, hires2: 'hires2', hires4: 'hires4',
        point_coords: { dims: [1, 2, 2], data: [0,0,0,0] }, point_labels: { dims: [1,2], data: [4,4], dtype: 'int64' } },
        outputs: { masks: `track-${n}-masks`, scores: `track-${n}-scores`, pointers: `track-${n}-pointers`, object_score: `track-${n}-object-score` } }))
];
const compare = (a, b) => {
    let max=0, sum=0, intersect=0, union=0;
    for(let i=0;i<a.length;i++) { const d=Math.abs(a[i]-b[i]);max=Math.max(max,d);sum+=d;if(a[i]>0&&b[i]>0)intersect++;if(a[i]>0||b[i]>0)union++; }
    return { max, mae:sum/a.length, signIoU:union?intersect/union:1 };
};
const sessions={};const cpu=[];
for(const task of tasks){
    sessions[task.part] ??= await ort.InferenceSession.create(`static/models/sam2-large/${manifest[task.part].file}`,{executionProviders:['cpu'],intraOpNumThreads:8});
    const inputs={};
    for(const[name, source]of Object.entries(task.inputs)){
        const info=typeof source==='string'?fixtures[source]:source;
        let data;
        if(typeof source==='string'){const buf=await readFile(`${root}/${source}.bin`);const array=buf.buffer.slice(buf.byteOffset,buf.byteOffset+buf.byteLength);data=info.dtype==='int64'?new BigInt64Array(array):new Float32Array(array);}
        else data=info.dtype==='int64'?BigInt64Array.from(info.data,BigInt):Float32Array.from(info.data);
        inputs[name]=new ort.Tensor(info.dtype==='int64'?'int64':'float32',data,info.dims);
    }
    const outputs=await sessions[task.part].run(inputs);
    for(const[name,expected]of Object.entries(task.outputs)){
        const buf=await readFile(`${root}/${expected}.bin`);const reference=new Float32Array(buf.buffer.slice(buf.byteOffset,buf.byteOffset+buf.byteLength));
        const result={task:task.name,output:name,...compare(outputs[name].data,reference)};cpu.push(result);console.log('CPU',result);
    }
}
for(const session of Object.values(sessions))await session.release();
const browser=await chromium.launch({executablePath:process.env.SAGO_CHROME||'/usr/bin/google-chrome',args:['--enable-unsafe-webgpu','--ignore-gpu-blocklist','--use-angle=vulkan','--enable-features=Vulkan','--disable-vulkan-surface']});
const page=await browser.newPage();
page.on('console',m=>{if(m.type()==='error')console.log(m.text());});
await page.route('**/__memory_fixture/*',async route=>{const file=new URL(route.request().url()).pathname.split('/').pop();await route.fulfill({body:await readFile(`${root}/${file}`)});});
let gpu;
try{
    await page.goto('http://127.0.0.1:3000');
    gpu=await page.evaluate(async({fixtures,manifest,tasks})=>{
        const ort=await import('/static/lib/ort/ort.webgpu.bundle.min.mjs');ort.env.wasm.wasmPaths=new URL('/static/lib/ort/',location.href).href;ort.env.wasm.numThreads=1;
        const bytes=async url=>new Uint8Array(await(await fetch(url)).arrayBuffer());
        const sessions={},results=[];
        for(const task of tasks){
            if(!sessions[task.part]){
                const part=manifest[task.part],externalData=[];
                for(const path of part.externalData)externalData.push({path,data:await bytes('/static/models/sam2-large/'+path)});
                sessions[task.part]=await ort.InferenceSession.create(await bytes('/static/models/sam2-large/'+part.file),{externalData,executionProviders:['webgpu']});
            }
            const inputs={};
            for(const[name,source]of Object.entries(task.inputs)){
                const info=typeof source==='string'?fixtures[source]:source;
                const raw=typeof source==='string'?(await bytes('/__memory_fixture/'+source+'.bin')).buffer:null;
                const data=info.dtype==='int64'?(raw?new BigInt64Array(raw):BigInt64Array.from(info.data,BigInt)):(raw?new Float32Array(raw):Float32Array.from(info.data));
                inputs[name]=new ort.Tensor(info.dtype==='int64'?'int64':'float32',data,info.dims);
            }
            const device=await ort.env.webgpu.device;device.pushErrorScope('validation');
            const outputs=await sessions[task.part].run(inputs);
            const error=await device.popErrorScope();if(error)throw new Error(error.message);
            for(const[name,expected]of Object.entries(task.outputs)){
                const reference=new Float32Array((await bytes('/__memory_fixture/'+expected+'.bin')).buffer),actual=outputs[name].data;
                let max=0,sum=0,intersect=0,union=0;
                for(let i=0;i<actual.length;i++){const d=Math.abs(actual[i]-reference[i]);max=Math.max(max,d);sum+=d;if(actual[i]>0&&reference[i]>0)intersect++;if(actual[i]>0||reference[i]>0)union++;}
                results.push({task:task.name,output:name,max,mae:sum/actual.length,signIoU:union?intersect/union:1});
            }
            Object.values(inputs).forEach(t=>t.dispose());Object.values(outputs).forEach(t=>t.dispose());
        }
        for(const s of Object.values(sessions))await s.release();return results;
    },{fixtures,manifest,tasks});
}finally{await browser.close();}
for(const result of gpu)console.log('WebGPU',result);
await writeFile(`${root}/report.json`,JSON.stringify({cpu,gpu},null,2));
if([...cpu,...gpu].some(r=>!Number.isFinite(r.mae)||r.mae>0.02))throw new Error('Memory parity tolerance exceeded; inspect report.json');
