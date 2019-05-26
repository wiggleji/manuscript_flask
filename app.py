from flask import Flask, request, jsonify
from flask_cors import CORS
from manuscript_cpu import inference_one

app = Flask(__name__)

cors = CORS(app, resources={
    r"/*": {"origin": "*"},
})


@app.route('/', methods=['GET'])
def hello():
    sentence = request.args.get('text')
    print(sentence)
    result = inference_one(sentence)
    return jsonify({'result': result})


if __name__ == '__main__':
    app.run()
