import pickle
from flask import Flask,jsonify
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_distances
import jieba
app=Flask(__name__)
know = pickle.loads(open('knowledge.pickle','rb').read() )
vec = pickle.loads(open('vectorizer.pickle','rb').read() )
ctbc_qa = pickle.loads(open('data.pickle','rb').read() )


@app.route("/<user_input>")
def getResponse(user_input):
    c = [' '.join(jieba.cut(user_input))]
    user_vec = vec.transform(c)
    cs = cosine_distances(user_vec,know)
    idx = cs.argsort()[0][0]
    if cs[0][idx] < 0.5:
        res = ctbc_qa.iloc[cs.argsort()[0][0]]
        return jsonify({'question':res['question'], 'response':res['answer']})
    else:
        return jsonify({'question':'', 'resonse':'我現在無法回答您的問題，等我變聰明以後再回答您的問題'})

if __name__=="__main__":
    app.run(host='0.0.0.0', port=8080)
