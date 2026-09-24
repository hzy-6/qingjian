//! 把按 UTF-8 排好序的「字 n-gram<TAB>计数」流压成可 mmap 的 FST。

use std::fs::File;
use std::io::{self, BufRead};
use std::path::PathBuf;

use fst::MapBuilder;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .ok_or("usage: pack_char OUTPUT.fst < sorted-counts.tsv")?;
    let file = File::create(path)?;
    let mut builder = MapBuilder::new(file)?;
    let mut count = 0_usize;
    for line in io::stdin().lock().lines() {
        let line = line?;
        let (gram, frequency) = line.split_once('\t').ok_or("expected gram<TAB>count")?;
        builder.insert(gram, frequency.parse::<u64>()?)?;
        count += 1;
    }
    builder.finish()?;
    eprintln!("packed {count} character n-grams");
    Ok(())
}
