from model.learning_model import LearningModel


def from_model_to_bxon(learning_model: LearningModel):
    return {
        'model_id': learning_model.model_id,
        'processed': learning_model.processed,
        'w_vector': learning_model.w_vector,
        'd': learning_model.d
    }


def from_bson_to_model(bson):
    model_id = bson['model_id']
    processed = bson['processed']
    w_vector = bson['w_vector']
    d = bson['d']

    model = LearningModel(model_id, w_vector, d)
    model.processed = processed
    return model
