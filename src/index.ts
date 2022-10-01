import * as tf from '@tensorflow/tfjs';
import * as rvm from './rvm';
import { webcam } from './webcam';

const segmentationConfig: rvm.SegmentationConfig = {
  modelPath: '../models/mb3-i8/rvm.json',
  ratio: 0.5,
  mode: 'default',
};

const log = (...msg) => console.log(...msg); // eslint-disable-line no-console

async function main() {
  // gather dom elements
  const dom = {
    video: document.getElementById('video') as HTMLVideoElement,
    webcam: document.getElementById('webcam') as HTMLVideoElement,
    output: document.getElementById('output') as HTMLCanvasElement,
    merge: document.getElementById('merge') as HTMLCanvasElement,
    mode: document.getElementById('mode') as HTMLSelectElement,
    composite: document.getElementById('composite') as HTMLSelectElement,
    ratio: document.getElementById('ratio') as HTMLInputElement,
    fps: document.getElementById('fps') as HTMLDivElement,
  };
  // set defaults
  dom.fps.innerText = 'initializing';
  dom.ratio.valueAsNumber = segmentationConfig.ratio;
  dom.video.src = '../assets/rijeka.mp4';
  dom.composite.innerHTML = ['source-atop', 'color', 'color-burn', 'color-dodge', 'copy', 'darken', 'destination-atop', 'destination-in', 'destination-out', 'destination-over', 'difference', 'exclusion', 'hard-light', 'hue', 'lighten', 'lighter', 'luminosity', 'multiply', 'overlay', 'saturation', 'screen', 'soft-light', 'source-in', 'source-out', 'source-over', 'xor'].map((gco) => `<option value="${gco}">${gco}</option>`).join(''); // eslint-disable-line max-len
  const ctxMerge = dom.merge.getContext('2d') as CanvasRenderingContext2D;

  // initialize tfjs and load model
  await tf.setBackend('webgl');
  await tf.ready();
  tf.env().set('WEBGL_USE_SHAPES_UNIFORMS', true); // better tfjs performance when using webgl backend
  await rvm.load(segmentationConfig);
  log({ segmentationConfig });
  log({ tf: tf.version_core, backend: tf.getBackend(), state: tf.engine().state });
  const numTensors = tf.engine().state.numTensors;

  // initialize webcam
  dom.webcam.onplay = () => { // start processing on video play
    loop(); // eslint-disable-line no-use-before-define
    dom.output.width = webcam.width;
    dom.output.height = webcam.height;
    dom.merge.width = webcam.width;
    dom.merge.height = webcam.height;
  };
  await webcam.start({ element: dom.webcam, crop: true, width: 960, height: 720 });
  if (!webcam.track) dom.fps.innerText = 'webcam error';

  // processing loop
  async function loop() {
    if (!webcam.element || webcam.paused) return; // check if webcam is valid and playing
    const imageTensor = tf.browser.fromPixels(webcam.element); // read webcam frame
    const t0 = Date.now();
    segmentationConfig.mode = dom.mode.value as rvm.SegmentationMode; // get segmentation mode from ui
    segmentationConfig.ratio = dom.ratio.valueAsNumber; // get segmentation downsample ratio from ui
    const rgba = await rvm.predict(imageTensor as tf.Tensor3D, segmentationConfig); // run model and process results
    const t1 = Date.now();
    dom.fps.innerText = `fps: ${Math.round(10000 / (t1 - t0)) / 10}`; // mark performance
    tf.browser.toPixels(rgba, dom.output); // draw raw output
    ctxMerge.globalCompositeOperation = 'source-over';
    ctxMerge.drawImage(dom.video, 0, 0); // draw original video to first stacked canvas
    ctxMerge.globalCompositeOperation = dom.composite.value as GlobalCompositeOperation;
    ctxMerge.drawImage(dom.output, 0, 0); // draw processed output to second stacked canvas
    tf.dispose([imageTensor, rgba]); // dispose tensors
    if (numTensors !== tf.engine().state.numTensors) log({ leak: tf.engine().state.numTensors - numTensors }); // check for memory leaks
    requestAnimationFrame(loop);
  }
}

window.onload = main;
