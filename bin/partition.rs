use sentence2vec::prelude::*;

fn main() {
    env_logger::init();

    let word2vec: Word2Vec<300> = Word2Vec::load_from_bytes("cc.en.300.bin");
    word2vec.partition("./parts", 9000, 1000)
}
