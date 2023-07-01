# Sentence2Vec
This computes the sentence vector of a sentence using word2vec for each word in the sentence.  
The crate supports the txt format of word2vec and a custom format of word2vec for binary files.

## Usage
```rust
use sentence2vec::Sentence2Vec;

let sentence = "This is a sentence.";
let model = Sentence2Vec::new("path/to/model.bin").unwrap();
let vector = model.sentence_vector(sentence);
```
The sentence vector is a sum up of the word vectors of the sentence.
We then have a vector of the same dimension as the word vectors.

## Example
Here is an example to compute the cosine similarity between two sentences.
```rust
use sentence2vec::Sentence2Vec;

let word2vec = Word2Vec::<300>::load_from_bytes("tests/common.en.300.bin").await.unwrap();
let sentence2vec = Sentence2Vec::new(Box::new(word2vec));
let similarity = sentence2vec.cosine("This is a test.", "This is a test.").await.unwrap();
assert_eq!(similarity, 1.0);

let similarity_fruits = sentence2vec.cosine("I love apples.", "I love pears").await.unwrap();

let similarity_fuits_cars = sentence2vec.cosine("I love apples.", "I love cars").await.unwrap();

assert!(similarity_fruits > similarity_fuits_cars);
```

## License
MIT
```