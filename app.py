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
    if not sentence:
        return '로딩중입니다'
    print(sentence)
    result = inference_one(sentence)
    return jsonify({'result': result})


@app.route('/limer_html', methods=['GET'])
def get_limer_html():
    sentence = request.args.get('text')
    html = limer_html(sentence)
    html = html.replace('overflow:scroll', 'overflow:hidden')
    return html


if __name__ == '__main__':
    app.run()
