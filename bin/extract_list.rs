
#[cfg(feature = "partition")]
#[tokio::main(flavor = "current_thread")]
async fn main() {
    use std::io::{BufRead, self, Write};

    use sentence2vec::prelude::*;

    env_logger::init();

    // Ask the binary to load the model from a text file.
    print!("Model path (.bin formatted by this program): ");
    io::stdout().flush().unwrap();
    let path = std::io::stdin().lock().lines().next().unwrap().unwrap();

    print!("Word list: ");
    io::stdout().flush().unwrap();
    let wordlist = std::io::stdin().lock().lines().next().unwrap().unwrap();

    print!("Target directory: ");
    io::stdout().flush().unwrap();
    let dist = std::io::stdin().lock().lines().next().unwrap().unwrap();
    
    let word2vec: Word2Vec<300> = Word2Vec::load_from_bytes(path).await;

    let subset = word2vec.get_subset_from_wordlist(wordlist).await;
    subset.save_to_bytes(dist).await;
}

#[cfg(not(feature = "partition"))]

fn main() {
    panic!("This binary is empty partition feature isn't set.");
}
