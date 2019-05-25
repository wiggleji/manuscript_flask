from flask import Flask, request, jsonify
from manuscript_cpu import inference_one

app = Flask(__name__)


@app.route('/', methods=['GET'])
def hello():
    sentence = request.args.get('text')
    print(sentence)
    result = inference_one(sentence)
    return jsonify({'result': result})


if __name__ == '__main__':
    app.run()
