from flask import Flask, request, render_template
import recommendation

app=Flask(__name__)

data = recommendation.load_data()
preprocessed_data = recommendation.preprocess_data(data)
transformed_data = recommendation.transform_data(preprocessed_data)

# main page
@app.route('/')
def welcome():
    return render_template('index.html')

# recommendation page
@app.route('/getrecommendation', methods=['GET','POST'])
def getrecommendation():
    if request.method == 'POST':
        title =request.form['title'].lower()
        if title in [i for i in data['title'].str.lower()]:
            result1 = recommendation.recommend_movies(title,preprocessed_data,transformed_data)
            isexist = "yes"
            return render_template('index.html',column_names=result1.columns.values, row_data=list(result1.values.tolist()), zip=zip, isexist=isexist, title= title.capitalize())
        else:
            result2 = ''
            isexist = "no"
            return render_template('index.html', result2= result2, isexist= isexist, title= title.capitalize())




if __name__=='__main__':
    app.debug=True
    app.run()