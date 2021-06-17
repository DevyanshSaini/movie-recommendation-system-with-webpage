from flask import Flask, render_template, request
from main import pas  

app=Flask(__name__)

@app.route('/')
def man():
    return render_template('home.html')

@app.route('/home', methods=['POST'])
def home():
    temp=request.form['name']
    ans=pas(temp)
    return render_template('home.html',data=ans)

if __name__ == "__main__":
    app.run(debug=True)  