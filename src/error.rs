use crate::tokenizing::TokenLocation;

pub fn error(line_span: (u32, u32), col_span: (u32, u32), source: &str) -> String {
    let lines = source.lines().collect::<Vec<_>>();
    let first_line = line_span.0 - 1;
    let last_line = line_span.1 - 1;
    let pre_first_line = first_line.checked_sub(1);
    let post_last_line = last_line + 1;
    let max_line_num_len = post_last_line.to_string().len();

    let mut string = String::new();

    string.push_str(&format!(
        "\x1b[31merror\x1b[0m: unexpected token at {}:{}\n",
        line_span.0, col_span.0
    ));

    if let Some(idx) = pre_first_line {
        let line = lines[idx as usize];
        let line_num = idx + 1;
        let line_num_str = line_num.to_string();
        dbg!(&line_num_str);
        dbg!(max_line_num_len);
        dbg!(post_last_line);
        dbg!(&line_span);
        let padding = max_line_num_len - line_num_str.len();
        string.push_str(&format!(
            " \x1b[34m{}{} | \x1b[0m",
            " ".repeat(padding),
            line_num_str
        ));
        string.push_str(line);
        string.push('\n');
    }

    for i in first_line..=last_line {
        let line = lines[i as usize];
        let line_num = i + 1;
        let line_num_str = line_num.to_string();
        let padding = max_line_num_len - line_num_str.len();
        let col_start = if i == first_line {
            col_span.0 - 1
        } else {
            0
        };
        let col_end = if i == last_line {
            col_span.1 - 1
        } else {
            line.len() as u32
        };
        let mut caret = String::new();
        for _ in 0..col_start {
            caret.push(' ');
        }
        for _ in col_start..col_end {
            caret.push('^');
        }
        string.push_str(&format!(
            " \x1b[34m{}{} | \x1b[0m",
            " ".repeat(padding),
            line_num_str
        ));
        println!("{}", &line[col_start as usize..col_end as usize]);
        let colored_line = format!(
            "{}{}{}",
            &line[..col_start as usize],
            format!(
                "\x1b[31m{}\x1b[0m",
                &line[col_start as usize..col_end as usize]
            ),
            &line[col_end as usize..],
        );
        string.push_str(&colored_line);
        string.push('\n');
        string.push_str(&format!(
            " \x1b[34m{} | \x1b[0m",
            " ".repeat(max_line_num_len)
        ));
        string.push_str(&format!("\x1b[31m{}\x1b[0m\n", caret));
    }

    if post_last_line < lines.len() as u32 {
        let line = lines[post_last_line as usize];
        let line_num = post_last_line + 1;
        let line_num_str = line_num.to_string();
        let padding = max_line_num_len - line_num_str.len();
        string.push_str(&format!(
            " \x1b[34m{}{} | \x1b[0m",
            " ".repeat(padding),
            line_num_str
        ));
        string.push_str(line);
        string.push('\n');
    }

    string
}
