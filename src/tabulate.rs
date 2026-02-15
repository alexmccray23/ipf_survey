use ipf::{DenseMatrix, IpfMatrix};

use crate::survey::CodedSurvey;

/// Cross-tabulate a survey into a `DenseMatrix` seed for IPF.
///
/// Shape = `[var_0.levels, var_1.levels, ...]`.
/// Each cell accumulates the sum of base weights for records whose codes
/// match that cell's multi-index.
pub fn tabulate(survey: &CodedSurvey) -> DenseMatrix<f64> {
    let shape: Vec<usize> = survey.variables().iter().map(|v| v.levels).collect();
    let mut matrix = DenseMatrix::zeros(shape);

    for r in 0..survey.n_records() {
        let codes = survey.record_codes(r);
        let w = survey.base_weight(r);
        let current = matrix.get(codes);
        matrix.set(codes, current + w);
    }

    matrix
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::survey::Variable;
    use ipf::IpfMatrix;

    fn make_survey(codes: Vec<usize>, n_records: usize) -> CodedSurvey {
        let vars = vec![
            Variable {
                name: "age".into(),
                levels: 3,
                labels: None,
            },
            Variable {
                name: "gender".into(),
                levels: 2,
                labels: None,
            },
        ];
        CodedSurvey::from_flat_codes(vars, codes, n_records).unwrap()
    }

    #[test]
    fn two_var_counts() {
        // 3x2 table: age(3) x gender(2)
        // Records: (0,0), (0,1), (1,0), (1,0), (2,1), (2,1)
        let survey = make_survey(vec![0, 0, 0, 1, 1, 0, 1, 0, 2, 1, 2, 1], 6);
        let matrix = tabulate(&survey);

        assert_eq!(matrix.shape(), &[3, 2]);
        assert_eq!(matrix.get(&[0, 0]), 1.0); // young male
        assert_eq!(matrix.get(&[0, 1]), 1.0); // young female
        assert_eq!(matrix.get(&[1, 0]), 2.0); // middle male
        assert_eq!(matrix.get(&[1, 1]), 0.0); // middle female
        assert_eq!(matrix.get(&[2, 0]), 0.0); // senior male
        assert_eq!(matrix.get(&[2, 1]), 2.0); // senior female
    }

    #[test]
    fn weighted_counts() {
        let vars = vec![
            Variable {
                name: "x".into(),
                levels: 2,
                labels: None,
            },
            Variable {
                name: "y".into(),
                levels: 2,
                labels: None,
            },
        ];
        let survey = CodedSurvey::from_flat_codes_weighted(
            vars,
            vec![0, 0, 0, 1, 1, 0],
            3,
            vec![2.0, 0.5, 1.5],
        )
        .unwrap();
        let matrix = tabulate(&survey);

        assert_eq!(matrix.get(&[0, 0]), 2.0);
        assert_eq!(matrix.get(&[0, 1]), 0.5);
        assert_eq!(matrix.get(&[1, 0]), 1.5);
        assert_eq!(matrix.get(&[1, 1]), 0.0);
    }

    #[test]
    fn single_variable() {
        let vars = vec![Variable {
            name: "x".into(),
            levels: 3,
            labels: None,
        }];
        let survey = CodedSurvey::from_flat_codes(vars, vec![0, 1, 1, 2], 4).unwrap();
        let matrix = tabulate(&survey);

        assert_eq!(matrix.shape(), &[3]);
        assert_eq!(matrix.get(&[0]), 1.0);
        assert_eq!(matrix.get(&[1]), 2.0);
        assert_eq!(matrix.get(&[2]), 1.0);
    }

    #[test]
    fn all_same_cell() {
        let survey = make_survey(vec![1, 0, 1, 0, 1, 0], 3);
        let matrix = tabulate(&survey);

        assert_eq!(matrix.get(&[1, 0]), 3.0);
        // All other cells should be 0
        assert_eq!(matrix.get(&[0, 0]), 0.0);
        assert_eq!(matrix.get(&[0, 1]), 0.0);
        assert_eq!(matrix.get(&[1, 1]), 0.0);
        assert_eq!(matrix.get(&[2, 0]), 0.0);
        assert_eq!(matrix.get(&[2, 1]), 0.0);
    }
}
