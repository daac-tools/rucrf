#[inline(always)]
pub fn logsumexp(a: f64, b: f64) -> f64 {
    if a == f64::NEG_INFINITY && b == f64::NEG_INFINITY {
        return f64::NEG_INFINITY;
    }
    if a > b {
        a + (b - a).exp().ln_1p()
    } else {
        b + (a - b).exp().ln_1p()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_logsumexp_small_1() {
        let a = 0.5f64;
        let b = 2f64;
        let expected = (a.exp() + b.exp()).ln();
        let result = logsumexp(a, b);
        assert!((expected - result).abs() < f64::EPSILON);
    }

    #[test]
    fn test_logsumexp_small_2() {
        let a = 12f64;
        let b = 5f64;
        let expected = (a.exp() + b.exp()).ln();
        let result = logsumexp(a, b);
        assert!((expected - result).abs() < f64::EPSILON);
    }

    #[test]
    fn test_logsumexp_large_1() {
        let a = 1234f64;
        let b = 1232f64;
        // log(exp(1234) + exp(1232))
        // = log(exp(1232 + 2) + exp(1232 + 0))
        // = log(exp(1232) * (exp(2) + exp(0)))
        // = 1232 + log(exp(2) + 1)
        let expected = 1232.0 + (2f64.exp() + 1.0).ln();
        let result = logsumexp(a, b);
        assert!((expected - result).abs() < f64::EPSILON);

        // The following naive calculation fails
        let naive = (a.exp() + b.exp()).ln();
        assert!(naive.is_infinite());
    }

    #[test]
    fn test_logsumexp_large_2() {
        let a = 1230f64;
        let b = 1235f64;
        // log(exp(1230) + exp(1235))
        // = log(exp(1230 + 0) + exp(1230 + 5))
        // = log(exp(1230) * (exp(0) + exp(5)))
        // = 1230 + log(1 + exp(5))
        let expected = 1230.0 + (1.0 + 5f64.exp()).ln();
        let result = logsumexp(a, b);
        assert!((expected - result).abs() < f64::EPSILON);

        // The following naive calculation fails
        let naive = (a.exp() + b.exp()).ln();
        assert!(naive.is_infinite());
    }

    #[test]
    fn test_logsumexp_inf_1() {
        let a = f64::INFINITY;
        let b = 2.0;
        let expected = f64::INFINITY;
        let result = logsumexp(a, b);
        assert_eq!(expected, result);
    }

    #[test]
    fn test_logsumexp_inf_2() {
        let a = f64::INFINITY;
        let b = f64::INFINITY;
        let result = logsumexp(a, b);
        assert!(result.is_nan());
    }

    #[test]
    fn test_logsumexp_inf_3() {
        let a = f64::NEG_INFINITY;
        let b = 2.0;
        let expected = 2.0;
        let result = logsumexp(a, b);
        assert!((expected - result).abs() < f64::EPSILON);
    }

    #[test]
    fn test_logsumexp_inf_4() {
        let a = f64::NEG_INFINITY;
        let b = f64::NEG_INFINITY;
        let result = logsumexp(a, b);
        assert_eq!(f64::NEG_INFINITY, result);
    }
}
