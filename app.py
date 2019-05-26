from flask import Flask, request, jsonify
from flask_cors import CORS
from manuscript_cpu import inference_one, limer_html

app = Flask(__name__)

cors = CORS(app, resources={
    r"/*": {"origin": "*"},
})


@app.route('/', methods=['GET'])
def hello():
    sentence = request.args.get('text')
    print(sentence)
    result = inference_one(sentence)
    html = limer_html(sentence)
    return jsonify({'result': result, 'html': html})


if __name__ == '__main__':
    app.run()
