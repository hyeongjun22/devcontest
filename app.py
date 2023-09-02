from flask import Flask, render_template
from flask import request
import main

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')




@app.route('/report')
def report():
    user = request.args.get('user')
    destination_latitude=request.args.get('destination_latitude')
    destination_longitude=request.args.get('destination_longitude')	
    schedule=request.args.get('schedule')	
    expense=request.args.get('expense')
    people=request.args.get('people')	
    age=request.args.get('age')	
    hometown_latitude=request.args.get('hometown_latitude')	
    hometown_longitude=request.args.get('hometown_longitude')	
    tendency_activity=request.args.get('tendency_activity')	
    tendency_preview=request.args.get('tendency_preview')	
    start_time=request.args.get('start_time')	
    finish_time=request.args.get('finish_time')
    li=main.result(float(destination_latitude),float(destination_longitude),int(schedule),int(expense),int(people),int(age),float(hometown_latitude),float(hometown_longitude),int(tendency_activity),int(tendency_preview),int(start_time),int(finish_time))
    print(li)
    if li==0:
        return render_template('report1.html', user=user)

    else:
        return render_template('report.html', user=user,data=li,num=int(people))

if __name__=="__main__":
    app.run(debug=True)
    # host 등을 직접 지정하고 싶다면
    # app.run(host="127.0.0.1", port="5000", debug=True)