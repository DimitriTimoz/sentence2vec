use std::{fs::File, path::Path, io::{self, BufRead}};


/// Read lines from a file and return an iterator over the lines.
pub(crate) fn read_lines<P>(filename: P) -> io::Result<io::Lines<io::BufReader<File>>> where P: AsRef<Path>, {
    let file = File::open(filename)?;
    Ok(io::BufReader::new(file).lines())
}