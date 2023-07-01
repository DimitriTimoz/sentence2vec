use sentence2vec::prelude::*;

#[test]
fn test_load() {
    /* let sentence2vec = Sentence2Vec::new(Word2Vec::<300>::load_from_bytes("word2vectest.bin"));
    let sentence = "I am a sentence";
    let vector = sentence2vec.get_vec(sentence);
    println!("{}: {:?}", sentence, vector) */
}

#[test]
fn test_convert_to_bin() {
    let sentence2vec = Word2Vec::<300>::load_from_txt("cc.en.300.vec");
    sentence2vec.save_to_bytes("cc.en.300.bin");
}
