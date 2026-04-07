//! Typed asset storage with generational handles.
//!
//! [`AssetStore<T>`] is a generational arena that stores assets of type `T` and
//! hands out [`Handle<T>`] tickets. Handles are lightweight (8 bytes), `Copy`,
//! and carry a generation counter so stale handles (pointing at a slot that has
//! been freed and reused) are detected at runtime instead of silently returning
//! the wrong asset.
//!
//! # Usage
//!
//! ```ignore
//! let mut meshes: AssetStore<GpuMesh> = AssetStore::new();
//! let h: Handle<GpuMesh> = meshes.add(my_mesh);
//! let mesh_ref: &GpuMesh = meshes.get(h);
//! ```
//!
//! The companion [`AssetManager`] bundles one store per asset type
//! (mesh, material, texture) for convenience.

use std::marker::PhantomData;

use crate::material::Material;
use crate::mesh::GpuMesh;
use crate::texture::Texture;

// ── Handle ──────────────────────────────────────────────────────────────

/// A typed, generational handle into an [`AssetStore`].
///
/// Handles are 8 bytes (`u32` index + `u32` generation), `Copy`, and
/// parameterised on the asset type `T` so you cannot accidentally pass a
/// mesh handle where a material handle is expected.
pub struct Handle<T> {
    index: u32,
    generation: u32,
    _marker: PhantomData<fn() -> T>,
}

// Manual impls to avoid requiring T: Clone/Copy/etc.
impl<T> Clone for Handle<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for Handle<T> {}

impl<T> PartialEq for Handle<T> {
    fn eq(&self, other: &Self) -> bool {
        self.index == other.index && self.generation == other.generation
    }
}

impl<T> Eq for Handle<T> {}

impl<T> std::hash::Hash for Handle<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.index.hash(state);
        self.generation.hash(state);
    }
}

impl<T> std::fmt::Debug for Handle<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Handle")
            .field("index", &self.index)
            .field("generation", &self.generation)
            .finish()
    }
}

impl<T> Handle<T> {
    /// Raw index (for advanced use only — prefer [`AssetStore::get`]).
    #[inline]
    pub fn index(self) -> u32 {
        self.index
    }

    /// Generation counter.
    #[inline]
    pub fn generation(self) -> u32 {
        self.generation
    }
}

// ── AssetStore ──────────────────────────────────────────────────────────

/// Slot in the generational arena.
enum Slot<T> {
    Occupied { value: T, generation: u32 },
    Vacant { generation: u32 },
}

/// Generational arena for assets of type `T`.
///
/// Freed slots are recycled via a free-list. Each slot tracks a generation
/// counter that increments on removal, so stale handles are caught.
pub struct AssetStore<T> {
    slots: Vec<Slot<T>>,
    free_list: Vec<u32>,
    len: usize,
}

impl<T> AssetStore<T> {
    /// Create an empty store.
    pub fn new() -> Self {
        Self {
            slots: Vec::new(),
            free_list: Vec::new(),
            len: 0,
        }
    }

    /// Number of live assets.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the store is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Insert an asset and return its handle.
    pub fn add(&mut self, value: T) -> Handle<T> {
        self.len += 1;

        if let Some(free_index) = self.free_list.pop() {
            let slot = &mut self.slots[free_index as usize];
            let generation = match slot {
                Slot::Vacant { generation } => *generation,
                Slot::Occupied { .. } => unreachable!("free-list pointed at occupied slot"),
            };
            *slot = Slot::Occupied { value, generation };
            Handle {
                index: free_index,
                generation,
                _marker: PhantomData,
            }
        } else {
            let index = self.slots.len() as u32;
            let generation = 0;
            self.slots.push(Slot::Occupied { value, generation });
            Handle {
                index,
                generation,
                _marker: PhantomData,
            }
        }
    }

    /// Get a shared reference to the asset, or `None` if the handle is stale.
    pub fn get(&self, handle: Handle<T>) -> Option<&T> {
        match self.slots.get(handle.index as usize)? {
            Slot::Occupied { value, generation } if *generation == handle.generation => Some(value),
            _ => None,
        }
    }

    /// Get a mutable reference to the asset, or `None` if the handle is stale.
    pub fn get_mut(&mut self, handle: Handle<T>) -> Option<&mut T> {
        match self.slots.get_mut(handle.index as usize)? {
            Slot::Occupied {
                value, generation, ..
            } if *generation == handle.generation => Some(value),
            _ => None,
        }
    }

    /// Remove the asset at `handle`, returning it if the handle is still valid.
    ///
    /// The slot's generation is bumped so any remaining copies of the handle
    /// become stale.
    pub fn remove(&mut self, handle: Handle<T>) -> Option<T> {
        let slot = self.slots.get_mut(handle.index as usize)?;
        match slot {
            Slot::Occupied { generation, .. } if *generation == handle.generation => {
                let next_gen = *generation + 1;
                let old = std::mem::replace(slot, Slot::Vacant { generation: next_gen });
                self.free_list.push(handle.index);
                self.len -= 1;
                match old {
                    Slot::Occupied { value, .. } => Some(value),
                    _ => unreachable!(),
                }
            }
            _ => None,
        }
    }

    /// Check whether a handle still refers to a live asset.
    pub fn contains(&self, handle: Handle<T>) -> bool {
        self.get(handle).is_some()
    }

    /// Iterate over all live assets with their handles.
    pub fn iter(&self) -> impl Iterator<Item = (Handle<T>, &T)> {
        self.slots.iter().enumerate().filter_map(|(i, slot)| {
            if let Slot::Occupied { value, generation } = slot {
                Some((
                    Handle {
                        index: i as u32,
                        generation: *generation,
                        _marker: PhantomData,
                    },
                    value,
                ))
            } else {
                None
            }
        })
    }
}

impl<T> Default for AssetStore<T> {
    fn default() -> Self {
        Self::new()
    }
}

// ── Convenience type aliases ────────────────────────────────────────────

/// Handle to a GPU mesh.
pub type MeshHandle = Handle<GpuMesh>;

/// Handle to a material.
pub type MaterialHandle = Handle<Material>;

/// Handle to a texture.
pub type TextureHandle = Handle<Texture>;

// ── AssetManager ────────────────────────────────────────────────────────

/// Bundles per-type asset stores for meshes, materials, and textures.
pub struct AssetManager {
    pub meshes: AssetStore<GpuMesh>,
    pub materials: AssetStore<Material>,
    pub textures: AssetStore<Texture>,
}

impl AssetManager {
    /// Create an empty asset manager.
    pub fn new() -> Self {
        Self {
            meshes: AssetStore::new(),
            materials: AssetStore::new(),
            textures: AssetStore::new(),
        }
    }
}

impl Default for AssetManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Handle properties ────────────────────────────────────────────

    #[test]
    fn handle_is_8_bytes() {
        assert_eq!(std::mem::size_of::<Handle<u32>>(), 8);
    }

    #[test]
    fn handle_debug_format() {
        let h: Handle<u32> = Handle {
            index: 1,
            generation: 2,
            _marker: PhantomData,
        };
        let s = format!("{h:?}");
        assert!(s.contains("index: 1"));
        assert!(s.contains("generation: 2"));
    }

    #[test]
    fn handle_clone_and_copy() {
        let h: Handle<u32> = Handle {
            index: 0,
            generation: 0,
            _marker: PhantomData,
        };
        let h2 = h;
        let h3 = h.clone();
        assert_eq!(h, h2);
        assert_eq!(h, h3);
    }

    #[test]
    fn handle_equality() {
        let a: Handle<u32> = Handle {
            index: 0,
            generation: 0,
            _marker: PhantomData,
        };
        let b: Handle<u32> = Handle {
            index: 0,
            generation: 1,
            _marker: PhantomData,
        };
        assert_ne!(a, b);
    }

    #[test]
    fn handle_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        let h: Handle<u32> = Handle {
            index: 0,
            generation: 0,
            _marker: PhantomData,
        };
        set.insert(h);
        assert!(set.contains(&h));
    }

    #[test]
    fn handle_index_and_generation() {
        let h: Handle<u32> = Handle {
            index: 5,
            generation: 3,
            _marker: PhantomData,
        };
        assert_eq!(h.index(), 5);
        assert_eq!(h.generation(), 3);
    }

    // ── AssetStore basics ────────────────────────────────────────────

    #[test]
    fn new_store_is_empty() {
        let store: AssetStore<i32> = AssetStore::new();
        assert!(store.is_empty());
        assert_eq!(store.len(), 0);
    }

    #[test]
    fn default_store_is_empty() {
        let store: AssetStore<i32> = AssetStore::default();
        assert!(store.is_empty());
    }

    #[test]
    fn add_increases_len() {
        let mut store = AssetStore::new();
        store.add(42);
        assert_eq!(store.len(), 1);
        assert!(!store.is_empty());
    }

    #[test]
    fn add_returns_valid_handle() {
        let mut store = AssetStore::new();
        let h = store.add(42);
        assert_eq!(h.index(), 0);
        assert_eq!(h.generation(), 0);
    }

    #[test]
    fn sequential_adds_get_sequential_indices() {
        let mut store = AssetStore::new();
        let a = store.add(1);
        let b = store.add(2);
        let c = store.add(3);
        assert_eq!(a.index(), 0);
        assert_eq!(b.index(), 1);
        assert_eq!(c.index(), 2);
        assert_eq!(store.len(), 3);
    }

    // ── get / get_mut ────────────────────────────────────────────────

    #[test]
    fn get_returns_inserted_value() {
        let mut store = AssetStore::new();
        let h = store.add(42);
        assert_eq!(store.get(h), Some(&42));
    }

    #[test]
    fn get_mut_allows_modification() {
        let mut store = AssetStore::new();
        let h = store.add(42);
        *store.get_mut(h).unwrap() = 99;
        assert_eq!(store.get(h), Some(&99));
    }

    #[test]
    fn get_returns_none_for_out_of_range_index() {
        let store: AssetStore<i32> = AssetStore::new();
        let fake: Handle<i32> = Handle {
            index: 100,
            generation: 0,
            _marker: PhantomData,
        };
        assert_eq!(store.get(fake), None);
    }

    // ── remove ───────────────────────────────────────────────────────

    #[test]
    fn remove_returns_value() {
        let mut store = AssetStore::new();
        let h = store.add(42);
        assert_eq!(store.remove(h), Some(42));
        assert_eq!(store.len(), 0);
    }

    #[test]
    fn stale_handle_returns_none_after_remove() {
        let mut store = AssetStore::new();
        let h = store.add(42);
        store.remove(h);
        assert_eq!(store.get(h), None);
    }

    #[test]
    fn double_remove_returns_none() {
        let mut store = AssetStore::new();
        let h = store.add(42);
        store.remove(h);
        assert_eq!(store.remove(h), None);
    }

    #[test]
    fn stale_handle_after_slot_reuse() {
        let mut store = AssetStore::new();
        let h1 = store.add(10);
        store.remove(h1);
        let h2 = store.add(20);
        // h1 and h2 share the same index but different generations.
        assert_eq!(h1.index(), h2.index());
        assert_ne!(h1.generation(), h2.generation());
        assert_eq!(store.get(h1), None); // stale
        assert_eq!(store.get(h2), Some(&20)); // valid
    }

    // ── contains ─────────────────────────────────────────────────────

    #[test]
    fn contains_live_handle() {
        let mut store = AssetStore::new();
        let h = store.add(42);
        assert!(store.contains(h));
    }

    #[test]
    fn contains_stale_handle() {
        let mut store = AssetStore::new();
        let h = store.add(42);
        store.remove(h);
        assert!(!store.contains(h));
    }

    // ── iter ─────────────────────────────────────────────────────────

    #[test]
    fn iter_empty_store() {
        let store: AssetStore<i32> = AssetStore::new();
        assert_eq!(store.iter().count(), 0);
    }

    #[test]
    fn iter_yields_all_live_assets() {
        let mut store = AssetStore::new();
        store.add(1);
        store.add(2);
        store.add(3);
        let values: Vec<&i32> = store.iter().map(|(_, v)| v).collect();
        assert_eq!(values.len(), 3);
        assert!(values.contains(&&1));
        assert!(values.contains(&&2));
        assert!(values.contains(&&3));
    }

    #[test]
    fn iter_skips_removed() {
        let mut store = AssetStore::new();
        let a = store.add(1);
        store.add(2);
        store.add(3);
        store.remove(a);
        let values: Vec<&i32> = store.iter().map(|(_, v)| v).collect();
        assert_eq!(values.len(), 2);
        assert!(values.contains(&&2));
        assert!(values.contains(&&3));
    }

    #[test]
    fn iter_handles_are_valid() {
        let mut store = AssetStore::new();
        store.add(10);
        store.add(20);
        for (handle, value) in store.iter() {
            assert_eq!(store.get(handle), Some(value));
        }
    }

    // ── Free-list recycling ──────────────────────────────────────────

    #[test]
    fn free_list_recycles_slots() {
        let mut store = AssetStore::new();
        let a = store.add(1);
        let b = store.add(2);
        store.remove(a);
        let c = store.add(3);
        // c should reuse a's slot (index 0).
        assert_eq!(c.index(), a.index());
        assert_eq!(c.generation(), 1);
        assert_eq!(store.len(), 2);
        assert_eq!(store.get(b), Some(&2));
        assert_eq!(store.get(c), Some(&3));
    }

    #[test]
    fn multiple_remove_and_reuse() {
        let mut store = AssetStore::new();
        let a = store.add(1);
        let b = store.add(2);
        let c = store.add(3);
        store.remove(a);
        store.remove(c);
        assert_eq!(store.len(), 1);

        let d = store.add(4);
        let e = store.add(5);
        assert_eq!(store.len(), 3);
        assert_eq!(store.get(b), Some(&2));
        assert_eq!(store.get(d), Some(&4));
        assert_eq!(store.get(e), Some(&5));
    }

    // ── AssetManager ─────────────────────────────────────────────────

    #[test]
    fn asset_manager_new_is_empty() {
        let am = AssetManager::new();
        assert!(am.meshes.is_empty());
        assert!(am.materials.is_empty());
        assert!(am.textures.is_empty());
    }

    #[test]
    fn asset_manager_default_is_empty() {
        let am = AssetManager::default();
        assert!(am.meshes.is_empty());
        assert!(am.materials.is_empty());
        assert!(am.textures.is_empty());
    }
}
