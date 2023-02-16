use std::ops::Index;

use super::fhr::FixedHeapArray;

pub struct Tensor<T> {
    data: FixedHeapArray<T>,
    dims: Vec<usize>
}

macro_rules! unwrap_or_eval {
    ($v:expr, $e:expr) => {
        match $v {
            Some(v) => v,
            _ => $e
        }
    };
}

impl<T: Clone> Tensor<T> {
    pub fn new(dims: Vec<usize>, default: T) -> Option<Tensor<T>> {
        let mut size: usize = 1;
        for dim in &dims {
            size = unwrap_or_eval!(size.checked_mul(*dim), return None);
        }

        let data = unwrap_or_eval!(FixedHeapArray::new(size, default), return None);

        return Some(Tensor { data, dims });
    }
}