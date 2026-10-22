//! Shared, fixed-state lexicographic fragment selection. The two internal meters
//! retain their own work/limit/error contracts; no external callbacks or sorting
//! allocation are accepted. Selection includes the FINAL full scan.

use serde_json::{Map, Value};
use std::cmp::Ordering;

pub(super) trait KeyMeter {
    type Error;
    fn candidate(&mut self) -> Result<(), Self::Error>;
    fn check_key(&self, key: &str) -> Result<(), Self::Error>;
    fn compare_keys(&mut self, a: &str, b: &str) -> Result<Ordering, Self::Error>;
}

pub(super) struct Selection<'a> {
    iterator: Option<serde_json::map::Iter<'a>>,
    selected: Option<(&'a String, &'a Value)>,
}

impl<'a> Selection<'a> {
    pub(super) const fn empty() -> Self {
        Self {
            iterator: None,
            selected: None,
        }
    }

    pub(super) fn run<M: KeyMeter>(
        &mut self,
        values: &'a Map<String, Value>,
        previous: Option<&str>,
        meter: &mut M,
    ) -> Result<(), M::Error> {
        self.selected = None;
        self.iterator = Some(values.iter());
        while let Some((key, value)) = self.iterator.as_mut().and_then(|iter| iter.next()) {
            meter.candidate()?;
            meter.check_key(key)?;
            if let Some(previous) = previous {
                if meter.compare_keys(key, previous)? != Ordering::Greater {
                    continue;
                }
            }
            if let Some((best, _)) = self.selected {
                if meter.compare_keys(key, best)? != Ordering::Less {
                    continue;
                }
            }
            self.selected = Some((key, value));
        }
        self.iterator = None;
        Ok(())
    }

    pub(super) fn selected(&self) -> Option<(&'a String, &'a Value)> {
        self.selected
    }

    #[cfg(test)]
    pub(super) fn iterator(&self) -> Option<&serde_json::map::Iter<'a>> {
        self.iterator.as_ref()
    }
}
