import * as tf from '@tensorflow/tfjs';

const options = {
  modelUrl: '../models/mb3-i8/rvm.json',
  resolution: [960, 640],
  downsampleRatio: 0.5,
};

const log = (...msg) => console.log(...msg); // eslint-disable-line no-console

async function drawMatte(fgr, pha, canvas: HTMLCanvasElement) {
  const rgba: tf.Tensor3D = tf.tidy(() => {
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
  tf.browser.toPixels(rgba, canvas);
  /*
  const pixelData = new Uint8ClampedArray(await rgba.data());
  const imageData = new ImageData(pixelData, width, height);
  const ctx = canvas.getContext('2d');
  if (ctx) ctx.putImageData(imageData, 0, 0);
  if (background) canvas.style.background = background;
  */
  rgba.dispose();
}

async function drawRecurrent(r, canvas: HTMLCanvasElement) {
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
  await tf.setBackend('webgl');
  await tf.ready();
  tf.env().set('WEBGL_USE_SHAPES_UNIFORMS', true);
  log({ tf: tf.version_core, backend: tf.getBackend() });

  const video = document.getElementById('video') as HTMLVideoElement;
  const canvas = document.getElementById('canvas') as HTMLCanvasElement;
  const select = document.getElementById('select') as HTMLSelectElement;
  const fps = document.getElementById('fps') as HTMLDivElement;
  video.width = options.resolution[0];
  video.height = options.resolution[1];
  const webcam = await tf.data.webcam(video, { });
  video.onclick = () => {
    if (video.paused) video.play();
    else video.pause();
  };
  video.onplay = () => {
    loop(); // eslint-disable-line no-use-before-define
  };
  // @ts-ignore
  log({ webcam: webcam?.stream?.getVideoTracks()?.[0], settings: webcam?.stream?.getVideoTracks()?.[0].getSettings() });

  const model = await tf.loadGraphModel(options.modelUrl);
  log({ model });

  let [r1i, r2i, r3i, r4i] = [tf.tensor(0.0), tf.tensor(0.0), tf.tensor(0.0), tf.tensor(0.0)]; // initialize recurrent state
  const downsampleRatio = tf.tensor(options.downsampleRatio); // set downsample ratio

  async function loop() { // inference loop
    if (video.paused) return;
    const img = await webcam.capture();
    const src = tf.tidy(() => img.expandDims(0).div(255)); // normalize input
    const t0 = Date.now();
    const [fgr, pha, r1o, r2o, r3o, r4o] = await model.executeAsync({ src, r1i, r2i, r3i, r4i, downsample_ratio: downsampleRatio }, ['fgr', 'pha', 'r1o', 'r2o', 'r3o', 'r4o']) as tf.Tensor[];
    const t1 = Date.now();
    fps.innerText = `fps: ${Math.round(10000 / (t1 - t0)) / 10}`;
    switch (select.value) {
      case 'none':
        drawMatte(fgr.clone(), pha.clone(), canvas);
        break;
      case 'alpha':
        drawMatte(null, pha.clone(), canvas);
        break;
      case 'foreground':
        drawMatte(fgr.clone(), null, canvas);
        break;
      case 'recurrent':
        drawRecurrent(r1o, canvas); // can use r10, r20, r3o, r4o
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
