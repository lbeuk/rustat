use std::mem::{ManuallyDrop, self};
use std::ptr::{NonNull, self};
use std::alloc::{self, Layout};
use std::ops::{Index, IndexMut, Deref, DerefMut};

fn main() {
    println!("Hello, world!");
}

struct TensorImpl<T> {
    ptr: NonNull<T>,
    size: usize
}

unsafe impl<T: Send> Send for TensorImpl<T> {}
unsafe impl<T: Sync> Sync for TensorImpl<T> {}

macro_rules! unwrap_or_eval {
    ( $e:expr, $else:expr) => {
        match $e {
            Ok(x) => x,
            Err(_) => $else
        }
    };
}

impl<T: Clone> TensorImpl<T> {
    pub fn new(size: usize, default: T) -> Option<Self> {
        if size <= 0 || size > isize::MAX.try_into().unwrap() {
            return None;
        }

        let layout = unwrap_or_eval!(Layout::array::<T>(size), return None);

        let ptr = unsafe { alloc::alloc(layout) };

        let ptr = match NonNull::new(ptr as *mut T) {
            Some(p) => p,
            None => alloc::handle_alloc_error(layout)
        };

        let mut new_t = TensorImpl {
            ptr,
            size
        };

        for t in &mut new_t {
            *t = default.clone();
        }   

        return new_t;

    }
}

impl<T> Deref for TensorImpl<T> {
    type Target =  [T];

    fn deref(&self) -> &Self::Target {
        return unsafe {std::slice::from_raw_parts(self.ptr.as_ptr(), self.size)};
    } 
}

impl<T> DerefMut for TensorImpl<T> {    
    fn deref_mut(&mut self) -> &mut [T] {
        return unsafe {std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.size)};
    }
}

pub struct IntoIter<T> {
    buf: NonNull<T>,
    cap: usize,
    start: *const T,
    end: *const T
}

impl<T> Iterator for IntoIter<T> {
    type Item = T;
    
    fn next(&mut self) -> Option<T> {
        if self.start == self.end {
            return None;
        }
        else {
            unsafe {
                let result = ptr::read(self.start);
                self.start = self.start.offset(1);
                return Some(result);
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = (self.end as usize - self.start as usize)
            / mem::size_of::<T>();
        return (len, Some(len));
    }
}

impl<T> IntoIterator for TensorImpl<T> {
    type Item = T;
    type IntoIter = IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        let tensor = ManuallyDrop::new(self);

        let ptr = tensor.ptr;
        let size = tensor.size;

        unsafe {
            IntoIter {
                buf: ptr,
                cap: size,
                start: ptr.as_ptr(),
                end: ptr.as_ptr().add(size)
                
            }
        }
    }


}

// impl<T> Index<usize> for TensorImpl<T> {
//     type Output = T;

//     fn index(&self, index: usize) -> &Self::Output {
        
//     }
// }
