const path = require('path');

main();

async function main() {
  const { pipeline } = await import('@xenova/transformers');
  const modelPath = path.resolve('../../assets/models/codegen-350M-mono');

  const pipe = await pipeline(
    'text-generation',
    modelPath,
    {
      isOVModel: true,
      model_file_name: 'openvino_model.xml',
    },
  );

  console.time('Output time:');
  const out = await pipe('def fib(n):', {
    'max_new_tokens': 100,
  });
  console.timeEnd('Output time:');

  console.log(out);
}
