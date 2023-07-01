
#[cfg(feature = "partition")]
fn main() {
    use std::io::{BufRead, self, Write};

    use sentence2vec::prelude::*;

    env_logger::init();

    // Ask the binary to load the model from a text file.
    print!("Model path (.bin formatted by this program): ");
    io::stdout().flush().unwrap();
    let path = std::io::stdin().lock().lines().next().unwrap().unwrap();

    print!("Target directory: ");
    io::stdout().flush().unwrap();
    let dist = std::io::stdin().lock().lines().next().unwrap().unwrap();
    
    print!("Number of partitions: ");
    io::stdout().flush().unwrap();
    let partitions = std::io::stdin().lock().lines().next().unwrap().unwrap().parse::<usize>().unwrap();

    print!("Number of folders: ");
    io::stdout().flush().unwrap();
    let folders = std::io::stdin().lock().lines().next().unwrap().unwrap().parse::<usize>().unwrap();

    let word2vec: Word2Vec<300> = Word2Vec::load_from_bytes(path);
    word2vec.partition(dist, partitions, folders)
}

#[cfg(not(feature = "partition"))]
fn main() {
    panic!("This binary is empty partition feature isn't set.");
}