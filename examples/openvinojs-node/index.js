const path = require('path');

const MODELS_DIR = '../../assets/models/';

main();

async function main() {
  const { pipeline } = await import('@xenova/transformers');
  // decicoder-1b-openvino-int8 also possible
  const modelPath = path.resolve(MODELS_DIR, 'codegen-350M-mono');

  const generation = await pipeline(
    'text-generation',
    modelPath,
    {
      isOVModel: true,
      'model_file_name': 'openvino_model.xml',
    },
  );

  console.time('Output time:');
  const out = await generation('def fib(n):', {
    'max_new_tokens': 100,
  });
  console.timeEnd('Output time:');

  console.log(out);
}
