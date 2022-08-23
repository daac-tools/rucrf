#[cfg(feature = "train")]
macro_rules! hashmap {
    ( $($k:expr => $v:expr,)* ) => {
        {
            #[allow(unused_mut)]
            let mut h = HashMap::new();
            $(
                h.insert($k, $v);
            )*
            h
        }
    };
    ( $($k:expr => $v:expr),* ) => {
        hashmap![$( $k => $v, )*]
    };
}

#[cfg(feature = "train")]
macro_rules! assert_alpha_beta {
    ( $expected:expr, $result:expr ) => {
        for (i, es) in $expected.iter().enumerate() {
            let mut es = es.clone();
            es.sort_unstable_by_key(|&(a, b, _)| (a, b));
            for (j, e) in es.iter().enumerate() {
                assert_eq!(e.0, $result[i][j].0);
                assert_eq!(e.1, $result[i][j].1);
                assert!((e.2 - $result[i][j].2).abs() < f64::EPSILON);
            }
        }
    };
}

#[cfg(feature = "train")]
pub(crate) use assert_alpha_beta;
#[cfg(feature = "train")]
pub(crate) use hashmap;
