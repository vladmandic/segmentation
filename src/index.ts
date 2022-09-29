import * as tf from '@tensorflow/tfjs';
import * as rvm from './rvm';
import { webcam } from './webcam';

const segmentationConfig: rvm.SegmentationConfig = {
  modelPath: '../models/mb3-i8/rvm.json',
  ratio: 0.5,
  mode: 'default',
}

const log = (...msg) => console.log(...msg); // eslint-disable-line no-console

async function main() {
  const video = document.getElementById('video') as HTMLVideoElement;
  const canvas = document.getElementById('canvas') as HTMLCanvasElement;
  const select = document.getElementById('select') as HTMLSelectElement;
  const ratio = document.getElementById('ratio') as HTMLInputElement;
  const fps = document.getElementById('fps') as HTMLDivElement;
  fps.innerText = 'initializing';

  await tf.setBackend('webgl');
  await tf.ready();
  tf.env().set('WEBGL_USE_SHAPES_UNIFORMS', true);
  log({ tf: tf.version_core, backend: tf.getBackend() });

  const model = await rvm.load(segmentationConfig);
  log({ model });

  video.onplay = () => { // start processing on video play
    loop(); // eslint-disable-line no-use-before-define
  };
  await webcam.start(video, { crop: true, width: 960, height: 720 });
  if (!webcam.track) fps.innerText = 'webcam error';

  const numTensors = tf.engine().state.numTensors;
  async function loop() { // inference loop
    if (!webcam.element || webcam.paused) return;
    const imageTensor = tf.browser.fromPixels(webcam.element);
    const t0 = Date.now();
    segmentationConfig.mode = select.value as rvm.SegmentationMode; // get mode from ui
    segmentationConfig.ratio = ratio.valueAsNumber; // get downsample ratio from ui
    const rgba = await rvm.predict(imageTensor as tf.Tensor3D, segmentationConfig); // run model and process results
    const t1 = Date.now();
    fps.innerText = `fps: ${Math.round(10000 / (t1 - t0)) / 10}`; // mark performance
    tf.browser.toPixels(rgba, canvas);
    tf.dispose([imageTensor, rgba]);
    if (numTensors !== tf.engine().state.numTensors) log({ leak: tf.engine().state.numTensors - numTensors }); // check for memory leaks
    requestAnimationFrame(loop);
  }
}

window.onload = main;
