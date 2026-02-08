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

  function getTensorRank(tensor) {
    if (tensor instanceof jwArray.Type) tensor = tensor.array;
    if (!Array.isArray(tensor)) return 0;
    return 1 + getTensorRank(tensor[0]);
  }

  function reshapeTensor(tensor, shape) {
    const flat = tensor.flat(Infinity).array;

    let size = 1;
    for (let i = 0; i < shape.length; i++) size *= shape[i];

    flat.length = size; // truncates or extends with empty slots
    for (let i = 0; i < size; i++) if (flat[i] === undefined) flat[i] = null;

    let idx = 0;
    const build = d => {
    const len = shape[d], arr = new Array(len);
    if (d === shape.length - 1) {
        for (let i = 0; i < len; i++) arr[i] = flat[idx++];
      } else {
        for (let i = 0; i < len; i++) arr[i] = build(d + 1);
      }
      return arr;
    };

    return build(0);
  }

  function fillTensor(tensor, val) {
    if (!Array.isArray(tensor) && !(tensor instanceof jwArray.Type)) return val;
    if (tensor instanceof jwArray.Type) tensor = tensor.array;
    return tensor.map(el => fillTensor(el instanceof jwArray.Type ? el.array : el, val));
  }

  function getTensorPath(tensor, path) {
    let current = tensor;
  
    for (let i = 0, len = path.length; i < len; i++) {
      if (current instanceof jwArray.Type) {
        current = current.array;
      } else if (!Array.isArray(current)) {
        return '';
      }
      current = current[path[i]-1];
      if (current === undefined) return '';
    }
  
    return current instanceof jwArray.Type ? current.array : current;
  }

  function setTensorPath(tensor, path, value) {
    function f(a, d) {
      if (!Array.isArray(a)) return;
      const i = path[d] - 1;
      if (i < 0 || i >= a.length) return;
  
      const c = a.slice();
      if (d === path.length - 1) {
        c[i] = value;
      } else {
        const n = c[i];
        const r = f(n instanceof jwArray.Type ? n.array : n, d + 1);
        if (r === undefined) return;
        c[i] = r;
      }
      return c;
    }
  
    const out = f(tensor.array, 0);
    return out === undefined ? '' : new jwArray.Type(out);
  }

  function validTensor(arr) {
    const u = x => { while (x instanceof jwArray.Type) x = x.array; return x; };
  
    let level = u(arr);
    if (!Array.isArray(level)) return false;
  
    while (true) {
      const len = level.length;
      if (len === 0) return false;
  
      let first = u(level[0]);
      const firstIsArray = Array.isArray(first);
      const expectedLen = firstIsArray ? first.length : 0;
  
      for (let i = 1; i < len; i++) {
        let item = u(level[i]);
        const isArr = Array.isArray(item);
        if (isArr !== firstIsArray) return false;
        if (isArr && item.length !== expectedLen) return false;
      }
  
      if (!firstIsArray) return true;
      level = first;
    }
  }

  function countScalars(t) {
    const u = x => { while (x instanceof jwArray.Type) x = x.array; return x; };
  
    function f(n) {
      n = u(n);
      if (!Array.isArray(n)) return 1;
  
      let total = 0;
      for (let i = 0, len = n.length; i < len; i++) {
        total += f(n[i]);
      }
      return total;
    }
  
    return f(t);
  }

  function findTensorPath(tensor, target) {
    const stack = [{ node: tensor, path: [] }];
  
    while (stack.length) {
      const { node, path } = stack.pop();
      let current = node;
  
      while (current instanceof jwArray.Type) current = current.array;
  
      if (Array.isArray(current)) {
        for (let i = current.length - 1; i >= 0; i--) {
          stack.push({ node: current[i], path: [...path, i] });
        }
      } else {
        let val = node;
        while (val instanceof jwArray.Type) val = val.array;
        if (val === target) return path.map(el => el + 1);
      }
    }
  
    return [];
  }

  function tensorContains(tensor, target) {
    if (tensor instanceof jwArray.Type) tensor = tensor.array;
    if (Array.isArray(tensor)) {
      for (let i = 0; i < tensor.length; i++) {
        if (tensorContains(tensor[i], target)) return true;
      }
      return false;
    }
    return tensor === target;
  }

  function transposeTensor(t) {
    while (t instanceof jwArray.Type) t = t.array;
    if (!validTensor(t)) return [];
  
    const shape = getTensorShape(t);
    const r = shape.length;
    if (r < 2) return t;
  
    const ns = shape.slice().reverse();
    const idx = new Array(r);
  
    function build(d) {
      const len = ns[d], out = new Array(len);
  
      if (d === r - 1) {
        for (let i = 0; i < len; i++) {
          idx[d] = i;
          let cur = t;
          for (let k = 0; k < r; k++) {
            while (cur instanceof jwArray.Type) cur = cur.array;
            cur = cur[idx[r - 1 - k]];
          }
          while (cur instanceof jwArray.Type) cur = cur.array;
          out[i] = cur;
        }
      } else {
        for (let i = 0; i < len; i++) {
          idx[d] = i;
          out[i] = build(d + 1);
        }
      }
  
      return out;
    }
  
    return build(0);
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
        color1: "#fe6743",
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
            text: 'blank tensor of shape [SHA]',
            blockType: Scratch.BlockType.REPORTER,
            blockShape: Scratch.BlockShape.SQUARE,
            arguments: {
              SHA: {type: Scratch.ArgumentType.STRING, shape: Scratch.BlockShape.SQUARE},
            },
            forceOutputType: "Array",
          },
          '---',
          {
            opcode: 'tensorGetPath',
            text: 'get path [PAT] in tensor [TEN]',
            blockType: Scratch.BlockType.REPORTER,
            allowDropAnywhere: true,
            arguments: {
              PAT: {type: Scratch.ArgumentType.STRING, shape: Scratch.BlockShape.SQUARE},
              TEN: jwArray.Argument
            },
          },
          {
            opcode: 'tensorFindPath',
            text: 'path of [VAL] in tensor [TEN]',
            blockType: Scratch.BlockType.REPORTER,
            blockShape: Scratch.BlockShape.SQUARE,
            allowDropAnywhere: true,
            arguments: {
              VAL: {type: Scratch.ArgumentType.STRING, exemptFromNormalization: true},
              TEN: jwArray.Argument
            },
            forceOutputType: "Array",
          },
          {
            opcode: 'tensorHas',
            text: 'tensor [TEN] has [VAL]',
            blockType: Scratch.BlockType.BOOLEAN,
            arguments: {
              TEN: jwArray.Argument,
              VAL: {type: Scratch.ArgumentType.STRING, exemptFromNormalization: true},
            },
          },
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
          {
            opcode: 'tensorRank',
            text: 'rank of tensor [TEN]',
            blockType: Scratch.BlockType.REPORTER,
            arguments: {
              TEN: jwArray.Argument
            },
          },
          {
            opcode: 'tensorScalars',
            text: 'size of tensor [TEN]',
            blockType: Scratch.BlockType.REPORTER,
            arguments: {
              TEN: jwArray.Argument
            },
          },
          '---',
          {
            opcode: 'tensorSetPath',
            text: 'set path [PAT] in tensor [TEN] to [VAL]',
            blockType: Scratch.BlockType.REPORTER,
            blockShape: Scratch.BlockShape.SQUARE,
            allowDropAnywhere: true,
            arguments: {
              PAT: {type: Scratch.ArgumentType.STRING, shape: Scratch.BlockShape.SQUARE},
              TEN: jwArray.Argument,
              VAL: {type: Scratch.ArgumentType.STRING, exemptFromNormalization: true}
            },
          },
          {
            opcode: 'tensorReshape',
            text: 'reshape tensor [TEN] to shape [SHA]',
            blockType: Scratch.BlockType.REPORTER,
            blockShape: Scratch.BlockShape.SQUARE,
            arguments: {
              TEN: jwArray.Argument,
              SHA: {type: Scratch.ArgumentType.STRING, shape: Scratch.BlockShape.SQUARE},
            },
            forceOutputType: "Array",
          },
          {
            opcode: 'tensorFill',
            text: 'fill tensor [TEN] with [VAL]',
            blockType: Scratch.BlockType.REPORTER,
            blockShape: Scratch.BlockShape.SQUARE,
            arguments: {
              TEN: jwArray.Argument,
              VAL: {type: Scratch.ArgumentType.STRING}
            },
            forceOutputType: "Array",
          },
          {
            opcode: 'tensorTranspose',
            text: 'transpose tensor [TEN]',
            blockType: Scratch.BlockType.REPORTER,
            blockShape: Scratch.BlockShape.SQUARE,
            arguments: {
              TEN: jwArray.Argument
            },
            forceOutputType: "Array",
          },
          '---',
          {
            opcode: 'tensorValid',
            text: 'is [TEN] a valid tensor?',
            blockType: Scratch.BlockType.BOOLEAN,
            arguments: {
              TEN: {type: Scratch.ArgumentType.STRING}
            },
          },
        ],
      };
    }

    blank() {
      return new jwArray.Type([], true);
    }
    blankSize({ SHA }) {
      SHA = jwArray.Type.toArray(SHA);
      if (SHA.array == null || (Array.isArray(SHA.array) && SHA.array.length === 0)) return new jwArray.Type([], true);
      return new jwArray.Type(createTensor(SHA.array));
    }

    tensorGetPath({ PAT, TEN }) {
      TEN = jwArray.Type.toArray(TEN);
      PAT = jwArray.Type.toArray(PAT);
      if (TEN.array == null || (Array.isArray(TEN.array) && TEN.array.length === 0)) return new jwArray.Type([], true);
      TEN = getTensorPath(TEN, PAT.array)
      return Array.isArray(TEN) ? new jwArray.Type(TEN) : TEN
    }
    tensorFindPath({ VAL, TEN }) {
      TEN = jwArray.Type.toArray(TEN);
      if (TEN.array == null || (Array.isArray(TEN.array) && TEN.array.length === 0)) return new jwArray.Type([], true);
      return new jwArray.Type(findTensorPath(TEN.array, VAL));
    }
    tensorHas({ TEN, VAL }) {
      TEN = jwArray.Type.toArray(TEN);
      if (TEN.array == null || (Array.isArray(TEN.array) && TEN.array.length === 0)) return new jwArray.Type([], true);
      return tensorContains(TEN, VAL)
    }
    tensorShape({ TEN }) {
      TEN = jwArray.Type.toArray(TEN);
      if (TEN.array == null || (Array.isArray(TEN.array) && TEN.array.length === 0)) return new jwArray.Type([], true);
      return new jwArray.Type(getTensorShape(TEN.array));
    }
    tensorRank({ TEN }) {
      TEN = jwArray.Type.toArray(TEN);
      if (TEN.array == null || (Array.isArray(TEN.array) && TEN.array.length === 0)) return '';
      return getTensorRank(TEN.array);
    }
    tensorScalars({ TEN }) {
      TEN = jwArray.Type.toArray(TEN);
      if (TEN.array == null || (Array.isArray(TEN.array) && TEN.array.length === 0)) return '';
      return countScalars(TEN.array);
    }

    tensorSetPath({ PAT, TEN, VAL }) {
      TEN = jwArray.Type.toArray(TEN);
      PAT = jwArray.Type.toArray(PAT);
      if (TEN.array == null || (Array.isArray(TEN.array) && TEN.array.length === 0)) return new jwArray.Type([], true);
      return setTensorPath(TEN, PAT.array, VAL)
    }
    tensorReshape({ TEN, SHA }) {
      TEN = jwArray.Type.toArray(TEN);
      SHA = jwArray.Type.toArray(SHA);
      if (TEN.array == null || (Array.isArray(TEN.array) && TEN.array.length === 0)) return new jwArray.Type([], true);
      return new jwArray.Type(reshapeTensor(TEN, SHA.array));
    }
    tensorFill({ TEN, VAL }) {
      TEN = jwArray.Type.toArray(TEN);
      if (TEN.array == null || (Array.isArray(TEN.array) && TEN.array.length === 0)) return new jwArray.Type([], true);
      return new jwArray.Type(fillTensor(TEN, VAL));
    }
    tensorTranspose({ TEN }) {
      TEN = jwArray.Type.toArray(TEN);
      if (TEN.array == null) return new jwArray.Type([], true);
      return new jwArray.Type(transposeTensor(TEN.array));
    }

    tensorValid({ TEN }) {
      TEN = jwArray.Type.toArray(TEN);
      const arr = TEN?.array;
      if (!Array.isArray(arr)) return false;
      return validTensor(arr);
    }
  }
  
  Scratch.extensions.register(new Extension());
})(Scratch);
