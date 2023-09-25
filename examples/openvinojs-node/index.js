const path = require('path');

const MODELS_DIR = '../../../';

main();

async function main() {
  const { pipeline } = await import('@xenova/transformers');
  // decicoder-1b-openvino-int8 also possible
  const modelPath = path.resolve(MODELS_DIR, 'decicoder-1b-openvino-int8');

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
    'callback_function': x => {
      console.log({
        output: generation.tokenizer.decode(x[0]['output_token_ids'])
      });
    }
  });
  console.timeEnd('Output time:');

  console.log(out);
}
