use std::mem::{ManuallyDrop, self};
use std::ptr::{NonNull, self};
use std::alloc::{self, Layout};
use std::ops::{Index, IndexMut, Deref, DerefMut};

/// FixedHeapArray provides arrays of known size at compile time. It can act
/// as a backend to many data structures such as matrices and tensors. It is
/// not meant to be used outside of the context of implementing another data
/// type.
pub struct FixedHeapArray<T> {
    ptr: NonNull<T>,
    size: usize
}

impl<T: Clone> FixedHeapArray<T> {
    pub fn new(size: usize, default: T) -> Option<Self> {
        if size <= 0 {
            return None;
        }

        // Implicitly checks maximum size for layout
        let layout = match Layout::array::<T>(size) {
            Ok(l) => l,
            _ => return None
        };

        let ptr = unsafe { alloc::alloc(layout) };

        let ptr = match NonNull::new(ptr as *mut T) {
            Some(p) => p,
            None => alloc::handle_alloc_error(layout)
        };

        let mut new_t = FixedHeapArray {
            ptr,
            size
        };

        for t in &mut new_t.iter_mut() {
            *t = default.clone();
        }   

        return Some(new_t);

    }
}

impl<T> Deref for FixedHeapArray<T> {
    type Target =  [T];

    fn deref(&self) -> &Self::Target {
        return unsafe {std::slice::from_raw_parts(self.ptr.as_ptr(), self.size)};
    } 
}

impl<T> DerefMut for FixedHeapArray<T> {    
    fn deref_mut(&mut self) -> &mut [T] {
        return unsafe {std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.size)};
    }
}

impl<T> IntoIterator for FixedHeapArray<T> {
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