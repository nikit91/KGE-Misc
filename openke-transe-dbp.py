# https://github.com/thunlp/OpenKE
import logging
import graphvite as gv
import graphvite.application as gap
#logging setup
logger = logging.getLogger("openke")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
# add the handlers to logger
logger.addHandler(ch)


def main():
    try:
        logger.info("Started")
        with open(gv.dataset.math.train, "r") as fin:
            for i in range(5):
                print(fin.readline().strip())

        app = gap.KnowledgeGraphApplication(dim=512)
        app.load(file_name=gv.dataset.math.train)
        app.build(optimizer=5e-3, num_negative=4)
        app.train(margin=9)

        app.link_prediction(file_name=gv.dataset.math.valid,
                            filter_files=[gv.dataset.math.train,
                                          gv.dataset.math.valid,
                                          gv.dataset.math.test],
                            target="tail")

        predictions = app.entity_prediction(file_name=gv.dataset.math.valid,
                                            target="tail", k=5)

        with open(gv.dataset.math.valid, "r") as fin:
            for i in range(5):
                print("ground truth: %s" % fin.readline().strip())
                print("top-5 prediction: %s" % ", ".join(["%s: %g" % x for x in predictions[i]]))
                print()

        entity2id = app.graph.entity2id
        relation2id = app.graph.relation2id
        entity_embeddings = app.solver.entity_embeddings
        relation_embeddings = app.solver.relation_embeddings

        # Get the entity embedding of "100"
        print(entity_embeddings[entity2id["100"]])

        app.save_model("rotate_math.pkl")

    except Exception as e:
        print(e)


if __name__ == '__main__':
    main()
