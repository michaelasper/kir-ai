use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvCacheBudget {
    max_tokens: usize,
    used_tokens: usize,
}

impl KvCacheBudget {
    pub fn new(max_tokens: usize) -> Self {
        Self {
            max_tokens,
            used_tokens: 0,
        }
    }

    pub fn max_tokens(&self) -> usize {
        self.max_tokens
    }

    pub fn used_tokens(&self) -> usize {
        self.used_tokens
    }

    pub fn remaining_tokens(&self) -> usize {
        self.max_tokens - self.used_tokens
    }

    pub fn reserve(&mut self, tokens: usize) -> Result<(), KvCacheError> {
        let next = self
            .used_tokens
            .checked_add(tokens)
            .ok_or(KvCacheError::CapacityExceeded {
                requested: tokens,
                available: self.remaining_tokens(),
            })?;
        if next > self.max_tokens {
            return Err(KvCacheError::CapacityExceeded {
                requested: tokens,
                available: self.remaining_tokens(),
            });
        }
        self.used_tokens = next;
        Ok(())
    }

    pub fn release(&mut self, tokens: usize) {
        self.used_tokens = self.used_tokens.saturating_sub(tokens);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvCacheError {
    CapacityExceeded { requested: usize, available: usize },
}

impl fmt::Display for KvCacheError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CapacityExceeded {
                requested,
                available,
            } => write!(
                formatter,
                "KV cache capacity exceeded: requested {requested} tokens, {available} available"
            ),
        }
    }
}

impl std::error::Error for KvCacheError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn budget_tracks_reserved_and_released_tokens() {
        let mut budget = KvCacheBudget::new(8);

        budget.reserve(3).expect("reserve fits");
        assert_eq!(budget.max_tokens(), 8);
        assert_eq!(budget.used_tokens(), 3);
        assert_eq!(budget.remaining_tokens(), 5);

        budget.release(2);
        assert_eq!(budget.used_tokens(), 1);
        assert_eq!(budget.remaining_tokens(), 7);
    }

    #[test]
    fn budget_rejects_over_capacity_reservation() {
        let mut budget = KvCacheBudget::new(4);
        budget.reserve(3).expect("first reserve fits");

        let err = budget.reserve(2).expect_err("capacity is enforced");

        assert_eq!(
            err,
            KvCacheError::CapacityExceeded {
                requested: 2,
                available: 1
            }
        );
        assert_eq!(budget.used_tokens(), 3);
    }

    #[test]
    fn release_saturates_at_zero() {
        let mut budget = KvCacheBudget::new(4);
        budget.reserve(1).expect("reserve fits");

        budget.release(99);

        assert_eq!(budget.used_tokens(), 0);
        assert_eq!(budget.remaining_tokens(), 4);
    }
}
