(function(Scratch) {
  'use strict';
  
  let jwArray = {
        Type: class { constructor(array) {/* noop */} static toArray(x) {/* noop */} },
        Block: {},
        Argument: {}
  }

  function createTensor(dims, val = null) {
    if (dims.length === 0) return val; 
    if (dims.some(el => typeof el !== 'number' || isNaN(el) || el <= 0)) return [];
    const [currentDim, ...remainingDims] = dims;
    return Array.from({ length: currentDim }, () => 
      createTensor(remainingDims, val)
    );
  }

  function getTensorShape(arr) {
    const shape = [];
    let level = arr;
  
    while (true) {
      const len = level.length;
      shape.push(len);
      if (len === 0) break;
  
      let first = level[0];
      if (first instanceof jwArray.Type) first = first.array;
  
      const firstIsArray = Array.isArray(first);
      const expectedLen = firstIsArray ? first.length : 0;
  
      for (let i = 1; i < len; i++) {
        let item = level[i];
        if (item instanceof jwArray.Type) item = item.array;
  
        const isArr = Array.isArray(item);
        if (isArr !== firstIsArray) return [];
        if (isArr && item.length !== expectedLen) return [];
      }
  
      if (!firstIsArray) break;
      level = first;
    }
  
    return shape;
  }
  
  class Extension {
    constructor() {
      if (!vm.jwArray) vm.extensionManager.loadExtensionIdSync('jwArray')
      jwArray = vm.jwArray
    }

    getInfo() {
      return {
        id: "lxTensors",
        name: "Tensors",
        color1: "#ff4f64",
        blocks: [
          {
            opcode: 'blank',
            text: 'blank tensor',
            blockType: Scratch.BlockType.REPORTER,
            blockShape: Scratch.BlockShape.SQUARE,
            forceOutputType: "Array",
            disableMonitor: true,
          },
          {
            opcode: 'blankSize',
            text: 'blank tensor of dimensions [DIM]',
            blockType: Scratch.BlockType.REPORTER,
            blockShape: Scratch.BlockShape.SQUARE,
            arguments: {
              DIM: jwArray.Argument
            },
            forceOutputType: "Array",
          },
          '---',
          {
            opcode: 'tensorShape',
            text: 'shape of tensor [TEN]',
            blockType: Scratch.BlockType.REPORTER,
            blockShape: Scratch.BlockShape.SQUARE,
            arguments: {
              TEN: jwArray.Argument
            },
            forceOutputType: "Array",
          },
        ],
      };
    }

    blank() {
      return new jwArray.Type([], true);
    }
    blankSize({ DIM }) {
      DIM = jwArray.Type.toArray(DIM);
      if (DIM.array == null || (Array.isArray(DIM.array) && DIM.array.length === 0)) return new jwArray.Type([], true);
      return new jwArray.Type(createTensor(DIM.array));
    }
    tensorShape({ TEN }) {
      TEN = jwArray.Type.toArray(TEN);
      if (TEN.array == null || (Array.isArray(TEN.array) && TEN.array.length === 0)) return new jwArray.Type([], true);
      return new jwArray.Type(getTensorShape(TEN.array));
    }
  }
  
  Scratch.extensions.register(new Extension());
})(Scratch);
