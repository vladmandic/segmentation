import * as tf from '@tensorflow/tfjs';

/**
 * Segmentation mode controls values returned by `predict` method
 * - `default` returns RGBA 4-channel tensor
 * - `alpha` returns Alpha 1-channel tensor
 * - `foreground` returns RGBA internally calculated foreground states
 * - `state` returns internal recurrent states as transferred between frames
 */
export type SegmentationMode = 'default' | 'alpha' | 'foreground' | 'state';

/**
 * RVM Segmentation Model configuration
 */
export type SegmentationConfig = {
  modelPath: string,
  ratio: number,
  mode: SegmentationMode,
}

// internal state varaibles
let model: tf.GraphModel;
const outputNodes = ['fgr', 'pha', 'r1o', 'r2o', 'r3o', 'r4o'];
const t: Record<string, tf.Tensor> = {}; // contains input tensor and recurrent states
let ratio = 0;

function init(config: SegmentationConfig) {
  tf.dispose([t.r1i, t.r2i, t.r3i, t.r4i, t.downsample_ratio]);
  t.r1i = tf.tensor(0.0);
  t.r2i = tf.tensor(0.0);
  t.r3i = tf.tensor(0.0);
  t.r4i = tf.tensor(0.0);
  ratio = config.ratio;
  t.downsample_ratio = tf.tensor(config.ratio); // initialize downsample ratio
}

/**
 * Loads RVM graph model from location specified in `config.modelPath`
 * and initialized initial states
 *
 * @param config SegmentationConfig
 * @returns GraphModel
 */
export async function load(config: SegmentationConfig): Promise<tf.GraphModel> {
  model = await tf.loadGraphModel(config.modelPath);
  init(config);
  return model;
}

function getRGBA(fgr: tf.Tensor | null, pha: tf.Tensor | null): tf.Tensor3D { // gets rgba // either fgr or pha must be present
  const norm = (r: tf.Tensor) => tf.tidy(() => {
    const squeeze = tf.squeeze(r, ([0]));
    const mul = tf.mul(squeeze, 255);
    const cast = tf.cast(mul, 'int32');
    return cast as tf.Tensor3D;
  });
  const rgb = fgr
    ? norm(fgr) // normalize and use value
    : tf.fill([pha!.shape[1] || 0, pha!.shape[2] || 0, 3], 255, 'int32'); // eslint-disable-line @typescript-eslint/no-non-null-assertion
  const a = pha
    ? norm(pha) // normalize and use value
    : tf.fill([fgr!.shape[1] || 0, fgr!.shape[2] || 0, 1], 255, 'int32'); // eslint-disable-line @typescript-eslint/no-non-null-assertion
  const rgba = tf.concat([rgb, a], -1) as tf.Tensor3D;
  tf.dispose([rgb, a]);
  return rgba;
}

function getState(state: tf.Tensor): tf.Tensor3D { // gets internal recurrent states
  return tf.tidy(() => {
    const r: Record<string, tf.Tensor | tf.Tensor[]> = {};
    r.unstack = tf.unstack(state, -1);
    r.concat = tf.concat(r.unstack, 1);
    r.split = tf.split(r.concat, 4, 1);
    r.stack = tf.concat(r.split, 2);
    r.squeeze = tf.squeeze(r.stack, [0]);
    r.expand = tf.expandDims(r.squeeze, -1);
    r.add = tf.add(r.expand, 1);
    r.mul = tf.mul(r.add, 127.5);
    r.cast = tf.cast(r.mul, 'int32');
    r.tile = tf.tile(r.cast, [1, 1, 3]);
    r.alpha = tf.fill([r.tile.shape[0] || 0, r.tile.shape[1] || 0, 1], 255, 'int32');
    return tf.concat([r.tile, r.alpha], -1) as tf.Tensor3D;
  });
}
/**
 * Runs model prediction based and returns processed image as tensor
 * Note that execution speed is directly related to input image resolution
 *
 * @param tensor Input tensor representing RGB image [width, height, channels = 3] where width and height can be dynamic
 * @param config Controls model post-processing and return values
 * @returns Tensor as [width, height, channels] where channels can be 4 (full RGBA) or 1(alpha-only) depending on `config`
 */
export async function predict(tensor: tf.Tensor, config: SegmentationConfig): Promise<tf.Tensor3D> {
  const expand = tf.expandDims(tensor, 0);
  t.src = tf.div(expand, 255);
  if (ratio !== config.ratio) init(config); // reinitialize recurrent states if requested downsample ratio changed
  const [fgr, pha, r1o, r2o, r3o, r4o] = await model.executeAsync(t, outputNodes) as tf.Tensor[]; // execute model
  let rgba: tf.Tensor3D;
  switch (config.mode) {
    case 'default':
      rgba = getRGBA(fgr, pha);
      break;
    case 'alpha':
      rgba = getRGBA(null, pha);
      break;
    case 'foreground':
      rgba = getRGBA(fgr, null);
      break;
    case 'state':
      rgba = getState(r1o); // can view any internal recurrent state r10, r20, r3o, r4o
      break;
    default:
      rgba = tf.tensor(0);
  }
  tf.dispose([t.src, expand, fgr, pha, t.r1i, t.r2i, t.r3i, t.r4i]);
  [t.r1i, t.r2i, t.r3i, t.r4i] = [r1o, r2o, r3o, r4o]; // update recurrent states
  return rgba;
}
