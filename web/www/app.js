const worker = new Worker('worker.js');
const $ = (id) => document.getElementById(id);
const status = (text) => { $('status').textContent = text; };
let ready = false;
let pending = null;

const EXAMPLES = [
  'normal_1.jpeg', 'normal_2.jpeg', 'pneumonia_1.jpeg', 'pneumonia_2.jpeg',
];
for (const name of EXAMPLES) {
  const button = document.createElement('button');
  button.type = 'button';
  button.title = name.replace('.jpeg', '').replace('_', ' ');
  button.innerHTML = `<img src="examples/${name}" alt="${button.title}" loading="lazy">`;
  button.onclick = async () => run(await (await fetch(`examples/${name}`)).arrayBuffer());
  $('examples').append(button);
}

function run(bytes) {
  if (!ready) { pending = bytes; status('Model still loading…'); return; }
  status('Running…');
  worker.postMessage({ type: 'classify', bytes }, [bytes]);
}

worker.onmessage = ({ data }) => {
  if (data.type === 'ready') {
    ready = true;
    $('tick').style.left = `${data.meta.threshold * 100}%`;
    status(`Model ready (loaded in ${(data.loadMs / 1000).toFixed(1)} s). Pick an image.`);
    if (pending) { const bytes = pending; pending = null; run(bytes); }
  } else if (data.type === 'result') {
    const pneumonia = data.prob >= data.threshold;
    const canvas = $('out');
    canvas.width = canvas.height = data.size;
    canvas.getContext('2d').putImageData(new ImageData(data.overlay, data.size, data.size), 0, 0);
    $('verdict').textContent = pneumonia ? 'Pneumonia' : 'Normal';
    $('prob').textContent = `P(pneumonia) ${(data.prob * 100).toFixed(1)}%`;
    $('fill').style.width = `${data.prob * 100}%`;
    status(`Inference ${Math.round(data.ms)} ms, on your device. Threshold ${data.threshold.toFixed(3)}.`);
  } else {
    status(`Error: ${data.message}`);
  }
};
worker.postMessage({ type: 'init' });

const drop = $('drop');
const file = $('file');
file.onchange = async () => file.files[0] && run(await file.files[0].arrayBuffer());
drop.onkeydown = (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); file.click(); } };
drop.ondragover = (e) => { e.preventDefault(); drop.classList.add('over'); };
drop.ondragleave = () => drop.classList.remove('over');
drop.ondrop = async (e) => {
  e.preventDefault();
  drop.classList.remove('over');
  const f = e.dataTransfer.files[0];
  if (f) run(await f.arrayBuffer());
};
