from model.learning_model import LearningModel


def from_model_to_bson(learning_model: LearningModel):
    return {
        'model_id': learning_model.model_id,
        'processed': learning_model.processed,
        'w_vector': learning_model.w_vector,
        'generation': learning_model.generation,
        'd': learning_model.d
    }


def from_bson_to_model(bson):
    model_id = bson['model_id']
    processed = bson['processed']
    w_vector = bson['w_vector']
    d = bson['d']
    score = bson['score']
    generation = bson['generation']

    model = LearningModel(w_vector, d, model_id)
    model.generation = generation
    model.processed = processed
    model.score = score
    return model
