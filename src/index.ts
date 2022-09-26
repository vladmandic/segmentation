import * as tf from '@tensorflow/tfjs';

const modelUrl = '../models/rvm_mobilenetv3_tfjs_int8/model.json';
const resolution = [640, 480];

const log = (...msg) => console.log(...msg); // eslint-disable-line no-console

async function drawMatte(fgr, pha, canvas, background) {
  const rgba = tf.tidy(() => {
    const rgb = (fgr !== null)
      ? fgr.squeeze(0).mul(255).cast('int32')
      : tf.fill([pha.shape[1], pha.shape[2], 3], 255, 'int32');
    const a = (pha !== null)
      ? pha.squeeze(0).mul(255).cast('int32')
      : tf.fill([fgr.shape[1], fgr.shape[2], 1], 255, 'int32');
    return tf.concat([rgb, a], -1);
  });
  if (fgr) fgr.dispose();
  if (pha) pha.dispose();
  const [height, width] = rgba.shape.slice(0, 2);
  const pixelData = new Uint8ClampedArray(await rgba.data());
  const imageData = new ImageData(pixelData, width, height);
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');
  if (ctx) ctx.putImageData(imageData, 0, 0);
  if (background) canvas.style.background = background;
  rgba.dispose();
}

async function drawHidden(r, canvas) {
  const rgba = tf.tidy(() => {
    r = r.unstack(-1);
    r = tf.concat(r, 1);
    r = r.split(4, 1);
    r = tf.concat(r, 2);
    r = r.squeeze(0);
    r = r.expandDims(-1);
    r = r.add(1).mul(127.5).cast('int32');
    r = r.tile([1, 1, 3]);
    r = tf.concat([r, tf.fill([r.shape[0], r.shape[1], 1], 255, 'int32')], -1);
    return r;
  });
  const [height, width] = rgba.shape.slice(0, 2);
  const pixelData = new Uint8ClampedArray(await rgba.data());
  const imageData = new ImageData(pixelData, width, height);
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');
  if (ctx) ctx.putImageData(imageData, 0, 0);
  rgba.dispose();
}

async function main() {
  await tf.ready();
  log({ tf: tf.version_core, backend: tf.getBackend() });
  const video = document.getElementById('video') as HTMLVideoElement;
  const canvas = document.getElementById('canvas') as HTMLCanvasElement;
  const select = document.getElementById('select') as HTMLSelectElement;
  video.width = resolution[0];
  video.height = resolution[1];
  const webcam = await tf.data.webcam(video);
  log({ webcam });
  video.onclick = () => {
    if (video.paused) video.play();
    else video.pause();
  };
  video.onplay = () => {
    loop(); // eslint-disable-line no-use-before-define
  };

  const model = await tf.loadGraphModel(modelUrl);
  log({ model });
  let [r1i, r2i, r3i, r4i] = [tf.tensor(0.0), tf.tensor(0.0), tf.tensor(0.0), tf.tensor(0.0)]; // initialize recurrent state
  const downsampleRatio = tf.tensor(0.5); // set downsample ratio

  async function loop() { // inference loop
    if (video.paused) return;
    await tf.nextFrame();
    const img = await webcam.capture();
    if (!img) return;
    const src = tf.tidy(() => img.expandDims(0).div(255)); // normalize input
    const [fgr, pha, r1o, r2o, r3o, r4o] = await model.executeAsync({ src, r1i, r2i, r3i, r4i, downsample_ratio: downsampleRatio }, ['fgr', 'pha', 'r1o', 'r2o', 'r3o', 'r4o']) as tf.Tensor[];
    switch (select.value) {
      case 'recurrent':
        drawHidden(r1o, canvas); // can use r10, r20, r3o, r4o
        break;
      case 'remove background':
        drawMatte(fgr.clone(), pha.clone(), canvas, 'rgb(0, 0, 0)');
        break;
      case 'white':
        drawMatte(fgr.clone(), pha.clone(), canvas, 'rgb(255, 255, 255)');
        break;
      case 'green':
        drawMatte(fgr.clone(), pha.clone(), canvas, 'rgb(120, 255, 155)');
        break;
      case 'alpha':
        drawMatte(null, pha.clone(), canvas, 'rgb(0, 0, 0)');
        break;
      case 'foreground':
        drawMatte(fgr.clone(), null, canvas, null);
        break;
      default:
    }
    tf.dispose([img, src, fgr, pha, r1i, r2i, r3i, r4i]);
    [r1i, r2i, r3i, r4i] = [r1o, r2o, r3o, r4o]; // Update recurrent states.
    requestAnimationFrame(loop);
  }

  loop(); // start initial
}

window.onload = main;
