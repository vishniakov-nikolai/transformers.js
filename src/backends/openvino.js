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
  const sortedFilenames = sortFilenames(filenames);

  const modelFiles = [];
  for (const filename of sortedFilenames) {
      const file = await getModelFile(modelDir, filename, true, options);

      modelFiles.push(file);
  }

  if (!GLOBAL_CORE) {
    console.log('== Create new Core instance');
    GLOBAL_CORE = new ov.Core();
  }

  const model = modelFiles.length === 2
    ? await GLOBAL_CORE.readModel(modelFiles[0], modelFiles[1])
    : await GLOBAL_CORE.readModel(modelFiles[0]);
  console.log('== here')
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

function sortFilenames(files) {
    const isXMLorBIN = new RegExp(/^.+\.(xml|bin)$/i);
    const isIRModel = files.length === 2
        && files.filter(name => isXMLorBIN.test(name)).length === 2;

    if (!isIRModel) return files;

    const isXML = new RegExp(/^.+\.(xml)$/i);
    const isBIN = new RegExp(/^.+\.(bin)$/i);

    // Order is important
    const xml = files.find(isXML.test);
    const bin = files.find(isBIN.test);

    return [xml, bin];
}
