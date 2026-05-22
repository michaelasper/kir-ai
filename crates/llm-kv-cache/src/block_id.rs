use std::{
    fmt,
    sync::atomic::{AtomicU64, Ordering},
};

use crate::KvCacheError;

static NEXT_BLOCK_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BlockId(u64);

impl BlockId {
    pub const INVALID: Self = Self(0);

    pub const fn new(raw: u64) -> Option<Self> {
        if raw == 0 { None } else { Some(Self(raw)) }
    }

    pub const fn as_u64(self) -> u64 {
        self.0
    }

    pub const fn is_valid(self) -> bool {
        self.0 != 0
    }

    pub const fn is_invalid(self) -> bool {
        self.0 == 0
    }

    pub(crate) fn next() -> Result<Self, KvCacheError> {
        let raw = NEXT_BLOCK_ID
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                current
                    .checked_add(1)
                    .filter(|next| *next != Self::INVALID.as_u64())
            })
            .map_err(|_| KvCacheError::InvalidShape)?;
        Ok(Self(raw))
    }
}

impl fmt::Display for BlockId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}", self.0)
    }
}
