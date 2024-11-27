from flask import Blueprint, request, jsonify
from service.categories import classify_event 
import utils.logging as logger 

fetch_categories_blueprint = Blueprint('fetch_categories', __name__)

@fetch_categories_blueprint.route('/api/classify', methods=['POST'])
def classify():
    try:
        data = request.get_json()
        event_name = data.get('event_name')
        event_desc = data.get('event_description')

        if not event_name or not event_desc:
            return jsonify({"error": "Both 'eventName' and 'eventDesc' are required."}), 400
        
        description=f"{event_name} {event_desc}"
        result = classify_event(description)
        return jsonify({"data":result,"error":None })
    except Exception as e:
        logger.log_message(message=f"Exception while getting L1/L2 tags {e}", level="error")
        return jsonify({"data":None,"error":f"Exception while getting L1/L2 tags {e}" })
