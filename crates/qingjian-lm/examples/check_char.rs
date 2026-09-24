//! 本地对拍字符 FST 的词频和整句分数。

use std::path::Path;

use qingjian_lm::CharNgramModel;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args();
    let _program = args.next();
    let path = args.next().ok_or("usage: check_char MODEL.fst TEXT ...")?;
    let model = CharNgramModel::from_path(Path::new(&path))?;
    for text in args {
        println!("{text}\t{}", model.score_text(&text));
    }
    Ok(())
}
