# Person segmentation using temporal guidance

Most ML models process frames as independent images  
This one uses recurrent neural network to process videos with temporal memory  
As a result, its far more stable between frames and also achieves higher performance  

- Credit: <https://github.com/PeterL1n/RobustVideoMatting>

## Code

- `model/mb3-i8` actual **rvm** model converted to **tfjs graph model** and quantized to `uint8`
- `src/rvm.ts` main module that exposes two methods, `load` and `predict`  
  module behavior is controlled via configuration params passed to `predict` method
- `src/index.ts` demo app that uses `rvm`  
  actual canvas merging (overlays, etc.) is done using native `canvas` `globalCompositeOperation`
- `src/webcam.ts` helper class used by demo that deals with webcam initialization and control
