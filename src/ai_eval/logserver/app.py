from flask import Flask, request
import redis
import json

app = Flask(__name__)

REDIS_PORT = 6379
REDIS_HOST = 'localhost'

# Connect to Redis
r = redis.Redis(
    host=REDIS_HOST, 
    port=REDIS_PORT, 
    db=0, 
    decode_responses=True
)

@app.route("/write_point", methods=['POST'])
def write_point():

    point_name = request.args.get('pt')
    log_obj = request.get_json()
    
    # string: json string
    r.set(point_name, json.dumps(log_obj))

    return { point_name: log_obj }


@app.route("/read_point")
def read_point():

    point_name = request.args.get('pt')
    
    log_obj_ser = r.get(point_name)
    log_obj = json.loads(log_obj_ser)

    return { point_name: log_obj }

