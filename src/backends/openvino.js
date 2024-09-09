const { hrtime } = require('process');
const OpenVINONode = require('openvino-node');

const { Tensor } = require('../utils/tensor.js');
const { getModelFile } = require('../utils/hub.js');

const { addon: ov } = OpenVINONode;

let GLOBAL_CORE = null;

module.exports = { create: getWrappedOVModelByPath };

async function getWrappedOVModelByPath(modelDir, filename, options) {
  const device = options.device || 'AUTO';
  const filenames = Array.isArray(filename) ? filename : [filename];
  const modelFiles = [];

  for (const filename of filenames) {
      const file = await getModelFile(modelDir, filename, true, options);

      modelFiles.push(file);
  }

  if (!GLOBAL_CORE) {
    console.log('== Create new Core instance');
    GLOBAL_CORE = new ov.Core();
  }

  const model = await getModel(modelFiles);
  const inputNames = model.inputs.map(i => i.toString());
  const compiledModel = await GLOBAL_CORE.compileModel(model, device);
  const ir = compiledModel.createInferRequest();
  console.log('== Use device ', device);

  return {
      run: async (inputData) => {
          const inputKeys = Object.keys(inputData);
          const tensorsDict = inputKeys.reduce((obj, key) => {
            obj[key] = convertToOVTensor(inputData[key]);

            return obj;
          }, {});

          const startTime = hrtime.bigint();
          const result = await ir.inferAsync(tensorsDict);
          const endTime = hrtime.bigint();

          if (typeof options.inferenceCallback === 'function')
            options.inferenceCallback(endTime - startTime);

          const modifiedOutput = {};

          Object.keys(result).forEach(name => {
              const ovTensor = result[name];
              const type = parseOVPrecision(ovTensor.getElementType());
              const shape = ovTensor.getShape();

              modifiedOutput[name] = new Tensor(type, ovTensor.data, shape);
          });

          return modifiedOutput;
      },
      inputNames,
  };

  function convertToOVTensor(inputTensor) {
      const { dims, type, data } = inputTensor;

      return new ov.Tensor(precisionToOV(type), dims, data);
  }

  function precisionToOV(str) {
      switch(str) {
          case 'int64':
              return ov.element.i64;
          case 'float32':
              return ov.element.f32;
          case 'bool':
              return ov.element.u8;
          default:
              throw new Error(`Undefined precision: ${str}`);
      }
  }

  function parseOVPrecision(elementType) {
      switch(elementType) {
          case ov.element.i64:
              return 'int64';
          case ov.element.f32:
              return 'float32';
          default:
              throw new Error(`Undefined precision: ${elementType}`);
      }
  }
}

function getModel(files) {
    const count = files.length;
    const isXMLorBIN = new RegExp(/^.+\.(xml|bin)$/i);
    const isXML = new RegExp(/^.+\.(xml)$/i);
    const isBIN = new RegExp(/^.+\.(bin)$/i);
    const isIRModel = count === 2
        && files.filter(name => isXMLorBIN.test(name)).length === 2;

    if (!isIRModel)
        return count === 2
            ? GLOBAL_CORE.readModel(files[0], files[1])
            : GLOBAL_CORE.readModel(files[0]);

    // Order is important
    const xml = files.find(isXML.test);
    const bin = files.find(isBIN.test);

    return GLOBAL_CORE.readModel(xml, bin);
}
