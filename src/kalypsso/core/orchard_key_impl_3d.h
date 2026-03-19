// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

KALYPSSO_DISABLE_CONVERSION_WARNINGS_PUSH()
KALYPSSO_DISABLE_NVCC_WARNINGS_PUSH()

/**
 * \file orchard_key_impl_3d.h
 */
// ====================================================================
// ====================================================================
// ====================================================================
template <>
struct orchard_key_t<3> : public BitFieldInteger<key_t>
{
  orchard_key_t(const orchard_key_t &) = default;

  orchard_key_t(orchard_key_t &&) = default;

  orchard_key_t &
  operator=(const orchard_key_t &) = default;

  orchard_key_t &
  operator=(orchard_key_t &&) = default;

  static constexpr uint8_t ONE_U = 1;

  /** dimension */
  static constexpr uint8_t DIM = 3;

  /** number of bits used to encode outside status */
  static constexpr uint8_t OUTSIDE_BITWIDTH = 3;

  /** number of bits used to encode level */
  static constexpr uint8_t LEVEL_BITWIDTH = 4;

  /** number of bits used to encode morton octant */
  static constexpr uint8_t MORTON_OCTANT_BITWIDTH = 42;

  /** number of bits used to encode morton tree */
  static constexpr uint8_t MORTON_TREE_BITWIDTH = 15;

  /** bit offset to the first bit where outside status bits start */
  static constexpr uint8_t OUTSIDE_OFFSET = 0;

  /** bit offset to the first bit where level bits start : 0 + 3 = 3 */
  static constexpr uint8_t LEVEL_OFFSET = OUTSIDE_OFFSET + OUTSIDE_BITWIDTH;

  /** bit offset to the first bit where the morton octant bits start : 3 + 4 = 7 */
  static constexpr uint8_t MORTON_OCTANT_OFFSET = LEVEL_OFFSET + LEVEL_BITWIDTH;

  /** bit offset to the first bit where tree bits start : 7 + 42 = 49 */
  static constexpr uint8_t MORTON_TREE_OFFSET = MORTON_OCTANT_OFFSET + MORTON_OCTANT_BITWIDTH;

  /** bits for outside status */
  static constexpr uint64_t OUTSIDE_MASK = 0x0000000000000007;

  /** bit mask to extract LEVEL (6 bits right after outside status) */
  static constexpr uint64_t LEVEL_MASK = 0x0000000000000078;

  /** bit mask to extract LEVEL (the middle 42 bits) */
  static constexpr uint64_t MORTON_OCTANT_MASK = 0x0001FFFFFFFFFF80;

  /** bit mask to extract MORTON_TREE (the upper 15 bits) */
  static constexpr uint64_t MORTON_TREE_MASK = 0xFFFE000000000000;

  /** level is in range [0, MAX_LEVEL] */
  static constexpr uint16_t MAX_LEVEL = MORTON_OCTANT_BITWIDTH / 3;

  /** number of levels : level is in range 0 to MAX_LEVELS */
  static constexpr uint16_t NUM_LEVELS = MAX_LEVEL + 1;

  /** length in logical units of the largest (root) quadrant */
  static constexpr uint32_t ROOT_LENGTH = 1 << MAX_LEVEL;

  /** logical tree size */
  static constexpr uint32_t TREE_SIZE = 1 << MAX_LEVEL;

  /** bit mask to keep only relevant bits for checking tree size */
  static constexpr uint32_t TREE_SIZE_MASK = TREE_SIZE - 1;

  /** maximum number of  trees per direction */
  static constexpr uint16_t MAX_NB_TREES_PER_DIR = 1 << (MORTON_TREE_BITWIDTH / DIM);

  /** number of children */
  static constexpr uint32_t NB_CHILDREN = 1 << DIM;

  /** number of neighbors across a face at fine level */
  static constexpr uint32_t NB_FACE_NEIGHBORS_FINE = NB_CHILDREN / 2;

  /** number of neighbors across an edge at fine level */
  static constexpr uint32_t NB_EDGE_NEIGHBORS_FINE = 2;

  /** maximum number of neighbors */
  static constexpr uint32_t MAX_NUM_NEIGHBORS = NB_CHILDREN / 2;

  using BitFieldInteger::BitFieldInteger;
  DECLARE_BIT(is_touching_face_X, OUTSIDE_OFFSET)
  DECLARE_BIT(is_touching_face_Y, OUTSIDE_OFFSET + 1)
  DECLARE_BIT(is_touching_face_Z, OUTSIDE_OFFSET + 2)
  DECLARE_CASTED_FIELD(outside_status, OUTSIDE_OFFSET, OUTSIDE_BITWIDTH, uint8_t)
  DECLARE_CASTED_FIELD(level, LEVEL_OFFSET, LEVEL_BITWIDTH, uint8_t)
  DECLARE_CASTED_FIELD(morton_octant, MORTON_OCTANT_OFFSET, MORTON_OCTANT_BITWIDTH, uint64_t)
  DECLARE_CASTED_FIELD(morton_tree, MORTON_TREE_OFFSET, MORTON_TREE_BITWIDTH, uint16_t)

  // ====================================================================
  //! get the outside status of a given key / quadrant
  KOKKOS_INLINE_FUNCTION static bool
  is_outside(uint64_t const & key)
  {

    return is_touching_face_X(key) or is_touching_face_Y(key) or is_touching_face_Z(key);

  } // is_outside

  // ====================================================================
  //! return true only if key is outside and "touching" inside domain by face along given "dir"
  //! \note a corner outside is touching inside domain by 2 faces.
  KOKKOS_INLINE_FUNCTION static bool
  is_outside_dir(uint64_t const & key, Dir::dir_t dir)
  {

    return ((dir == Dir::X) and is_touching_face_X(key)) or
           ((dir == Dir::Y) and is_touching_face_Y(key)) or
           ((dir == Dir::Z) and is_touching_face_Z(key));

  } // is_outside_dir

  // ====================================================================
  //! modify outside status of a given key.
  KOKKOS_INLINE_FUNCTION static void
  set_outside_bit(uint64_t & key, Face::face_t const & face)
  {

    if (face == Face::XMIN or face == Face::XMAX)
      set_is_touching_face_X(key, true);
    else if (face == Face::YMIN or face == Face::YMAX)
      set_is_touching_face_Y(key, true);
    else if (face == Face::ZMIN or face == Face::ZMAX)
      set_is_touching_face_Z(key, true);

  } // set_outside_bit

  // ====================================================================
  //! change outside status. By resetting these bits, key becomes an inside key
  KOKKOS_INLINE_FUNCTION static void
  reset_outside_bits(uint64_t & key)
  {
    set_is_touching_face_X(key, false);
    set_is_touching_face_Y(key, false);
    set_is_touching_face_Z(key, false);
  }

  // // ====================================================================
  // /** extract Morton tree */
  // KOKKOS_INLINE_FUNCTION static uint16_t
  // get_tree_morton_index(uint64_t key)
  // {
  //   return (key & MORTON_TREE_MASK);
  // } // get_tree_morton_index

  // // ====================================================================
  // /** extract Morton quadrant */
  // KOKKOS_INLINE_FUNCTION
  // static uint64_t
  // get_octant_morton_index(uint64_t key)
  // {
  //   return (key & MORTON_OCTANT_MASK) >> MORTON_OCTANT_OFFSET;
  // } // get_octant_morton_index

  // // ====================================================================
  // /** extract level */
  // KOKKOS_INLINE_FUNCTION
  // static uint8_t
  // get_level(uint64_t key)
  // {
  //   return (key & LEVEL_MASK) >> LEVEL_OFFSET;
  // } // get_level

  // ====================================================================
  template <int direction>
  KOKKOS_INLINE_FUNCTION static uint16_t
  get_tree_coord(uint64_t key)
  {
    static_assert(direction == IX || direction == IY || direction == IZ,
                  "Only X,Y and Z directions are possible in 3D");

    auto tree_coord = morton_tree(key);

    // select direction
    tree_coord = tree_coord >> direction;

    // only 5 bits are significant (= 001001001001001)
    tree_coord &= 0x1249;

    tree_coord = (tree_coord ^ (tree_coord >> 2)) & 0x70c3;
    tree_coord = (tree_coord ^ (tree_coord >> 4)) & 0xf00f;
    tree_coord = (tree_coord ^ (tree_coord >> 8)) & 0x00ff;

    return static_cast<uint16_t>(tree_coord);

  } // get_tree_coord

  // ====================================================================
  KOKKOS_INLINE_FUNCTION static Kokkos::Array<uint16_t, DIM>
  get_tree_coords(uint64_t key)
  {
    return Kokkos::Array<uint16_t, DIM>{ get_tree_coord<IX>(key),
                                         get_tree_coord<IY>(key),
                                         get_tree_coord<IZ>(key) };
  } // get_tree_coords

  // ====================================================================
  /**
   * get octant x,y or z coordinate from an orchard key.
   *
   * \param[in] key the orchard key
   */
  template <int direction>
  KOKKOS_INLINE_FUNCTION static uint32_t
  get_octant_coord(uint64_t key)
  {
    static_assert(direction == IX || direction == IY || direction == IZ,
                  "Only X,Y and Z directions are possible in 3D");

    // extract bits corresponding to morton octant
    key = morton_octant(key);

    // select direction
    key = key >> direction;

    // only 14 bits are significant, so applying a 42 bits wide mask should be enough
    // key &= 0x249249249249;
    key &= 0x9249249249;

    // clang-format off
    key = (key ^ (key >>  2)) & 0x30c30c30c30c30c3;
    key = (key ^ (key >>  4)) & 0xf00f00f00f00f00f;
    key = (key ^ (key >>  8)) & 0x00ff0000ff0000ff;
    key = (key ^ (key >> 16)) & 0x00ff00000000ffff;
    key = (key ^ (key >> 32)) & 0x1fffff;
    // clang-format on

    return static_cast<uint32_t>(key);

  } // get_octant_coord

  // ====================================================================
  /**
   * get octant x,y or z coordinate from an orchard key.
   *
   * \param[in] key the orchard key
   */
  KOKKOS_INLINE_FUNCTION static Kokkos::Array<uint32_t, 3>
  get_octant_coords(uint64_t key)
  {
    return Kokkos::Array<uint32_t, 3>{ get_octant_coord<IX>(key),
                                       get_octant_coord<IY>(key),
                                       get_octant_coord<IZ>(key) };
  } // get_octant_coords

  // ====================================================================
  /**
   * Take an integer representing one of the tree coordinate, and modify its
   * binary representation as preliminary step before encoding morton key.
   *
   * \param[in] coord must be in range [0,31] (only 5 bits are significant per direction)
   *
   * \return an integer with interleaved 0 in between each bits of input.
   */
  KOKKOS_INLINE_FUNCTION static uint16_t
  split_morton_tree(uint16_t coord)
  {
    // extract the last 5 bits
    uint16_t x = coord & 0x001F;

    x = (x | x << 8) & 0xf00f;
    x = (x | x << 4) & 0x70c3;
    x = (x | x << 2) & 0x1249;

    return x;

  } // split_morton_tree

  // ====================================================================
  /**
   * Take an integer representing one of the octant coordinate, and modify its
   * binary representation as preliminary step before encoding morton key.
   *
   * \param[in] coord must be in range [0,2^14-1] (only 14 bits are significant per direction)
   *
   * the result is stored on a 64 bits integer because 3x14 bits = 42 bits (minimum)
   *
   * \return an integer with interleaved 0 in between each bits of input.
   */
  KOKKOS_INLINE_FUNCTION static uint64_t
  split_morton_octant(uint32_t coord)
  {
    // extract the last 14 bits
    uint64_t x = coord & 0x3FFF;

    // clang-format off
    x = (x | x << 32) &   0x1f00000000ffff;
    x = (x | x << 16) &   0x1f0000ff0000ff;
    x = (x | x <<  8) & 0x100f00f00f00f00f;
    x = (x | x <<  4) & 0x10c30c30c30c30c3;
    x = (x | x <<  2) & 0x1249249249249249;
    // clang-format on

    return x;

  } // split_morton_octant

  // ====================================================================
  /**
   * encode morton tree index from x,y,z coordinates
   * \param[in] treeCoord is an an array of tree coordinates in the p4est brick connectivity
   */
  KOKKOS_INLINE_FUNCTION static uint16_t
  encode_morton_tree(Kokkos::Array<uint16_t, 3> treeCoord)
  {
    uint16_t key = 0;
    // clang-format off
    key |= split_morton_tree(treeCoord[IX]) |
           split_morton_tree(treeCoord[IY]) << 1 |
           split_morton_tree(treeCoord[IZ]) << 2;
    // clang-format on

    return key;
  } // encode_morton_tree

  // ====================================================================
  /**
   * encode morton tree index from x,y coordinates
   * \param[in] tree_x is tree cartesian coordinate along X in the p4est brick connectivity
   * \param[in] tree_y is tree cartesian coordinate along Y in the p4est brick connectivity
   * \param[in] tree_z is tree cartesian coordinate along Z in the p4est brick connectivity
   */
  KOKKOS_INLINE_FUNCTION static uint16_t
  encode_morton_tree(uint16_t tree_x, uint16_t tree_y, uint16_t tree_z)
  {
    uint16_t key = 0;
    // clang-format off
    key |= split_morton_tree(tree_x)      |
           split_morton_tree(tree_y) << 1 |
           split_morton_tree(tree_z) << 2;
    // clang-format on

    return key;
  } // encode_morton_tree

  // ====================================================================
  /**
   * encode morton tree index from x,y,z coordinates
   * \param[in] treeCoord is an an array of tree coordinates in the p4est brick connectivity
   */
  KOKKOS_INLINE_FUNCTION static void
  encode_morton_tree(uint64_t & orchard_key, Kokkos::Array<uint16_t, 3> treeCoord)
  {
    uint16_t tree_key = 0;

    // clang-format off
    tree_key |= split_morton_tree(treeCoord[IX])      |
                split_morton_tree(treeCoord[IY]) << 1 |
                split_morton_tree(treeCoord[IZ]) << 2;
    // clang-format on

    set_morton_tree(orchard_key, tree_key);

  } // encode_morton_tree

  // ====================================================================
  /**
   * encode morton octant index from octant x,y,z coordinates
   * \param[in] octCoord must be in [0,2**14-1]
   */
  KOKKOS_INLINE_FUNCTION static uint64_t
  encode_morton_octant(Kokkos::Array<uint32_t, 3> octCoord)
  {
    uint64_t key = 0;

    // clang-format off
    key |= split_morton_octant(octCoord[IX])      |
           split_morton_octant(octCoord[IY]) << 1 |
           split_morton_octant(octCoord[IZ]) << 2;
    // clang-format on

    return key;
  } // encode_morton_octant

  // ====================================================================
  /**
   * encode morton octant index from octant x,y,z coordinates
   * \param[in] octCoord must be in [0,2**14-1]^3
   */
  KOKKOS_INLINE_FUNCTION static void
  encode_morton_octant(uint64_t & orchard_key, Kokkos::Array<uint32_t, 3> octCoord)
  {
    uint64_t key = 0;

    // clang-format off
    key |= split_morton_octant(octCoord[IX])      |
           split_morton_octant(octCoord[IY]) << 1 |
           split_morton_octant(octCoord[IZ]) << 2;
    // clang-format on

    set_morton_octant(orchard_key, key);

  } // encode_morton_octant

  // ====================================================================
  /**
   * encode orchard key from octant x,y,z coordinates, tree x,y,z coords and level
   * \param[in] octCoord must be in [0,2**14-1]^3
   *
   */
  KOKKOS_INLINE_FUNCTION static uint64_t
  encode_orchard(Kokkos::Array<uint16_t, 3> treeCoord,
                 Kokkos::Array<uint32_t, 3> octCoord,
                 uint8_t                    level,
                 uint8_t                    outside_status)
  {
    uint64_t key = 0;

    set_outside_status(key, outside_status);
    set_level(key, level);
    encode_morton_octant(key, octCoord);
    encode_morton_tree(key, treeCoord);

    return key;

  } // encode_orchard

  // ====================================================================
  /**
   * encode orchard key from octant x,y,z coordinates, tree x,y,z coords and level
   * \param[in] octCoord must be in [0,2**14-1]^3
   *
   * key is always zero initialized, the outside status bits will be zero (i.e. inside quadrant).
   */
  KOKKOS_INLINE_FUNCTION static uint64_t
  encode_orchard(Kokkos::Array<uint16_t, 3> treeCoord,
                 Kokkos::Array<uint32_t, 3> octCoord,
                 uint8_t                    level)
  {
    uint64_t key = 0;

    set_level(key, level);
    encode_morton_octant(key, octCoord);
    encode_morton_tree(key, treeCoord);

    return key;

  } // encode_orchard

  // ====================================================================
  /**
   * encode orchard key from octant x,y,z coordinates, tree id and level
   * \param[in] octCoord must be in [0,2**14-1]^3
   *
   * key is always zero initialized, the outside bit will be zero.
   */
  KOKKOS_INLINE_FUNCTION static uint64_t
  encode_orchard(uint16_t                   treeid,
                 Kokkos::Array<uint32_t, 3> octCoord,
                 uint8_t                    level,
                 uint8_t                    outside_status = 0)
  {
    uint64_t key = 0;

    set_outside_status(key, outside_status);
    set_level(key, level);
    encode_morton_octant(key, octCoord);
    set_morton_tree(key, treeid);

    return key;

  } // encode_orchard

  /**
   * Octant logical length from level.
   *
   * \input[in] an AMR level
   *
   * \return octant length
   */
  KOKKOS_INLINE_FUNCTION
  static uint32_t
  octantLength_from_level(uint8_t level)
  {
    return (1 << (MAX_LEVEL - level));
  } // octantLength_from_level

  /**
   * Octant logical length.
   *
   * \input[in] a orchard key
   *
   * \return octant length
   */
  KOKKOS_INLINE_FUNCTION
  static uint32_t
  octantLength(uint64_t orchard_key)
  {
    return octantLength_from_level(level(orchard_key));
  } // octantLength

  /**
   * Last offset returns the logical coordinate (x, y, or z)
   * of an octant that "touches" the tree border on right.
   *
   * Just as a reminder, an octant that "touches" the tree border on left, e.g. along x axis, will
   * have x logical coordinate equal to zero.
   *
   */
  KOKKOS_INLINE_FUNCTION
  static uint32_t
  last_offset(uint64_t orchard_key)
  {
    return ROOT_LENGTH - octantLength(orchard_key);
  } // last_offset

  KOKKOS_INLINE_FUNCTION
  static uint32_t
  last_offset(uint8_t level)
  {
    return ROOT_LENGTH - octantLength_from_level(level);
  } // last_offset

  /**
   * Get the orchard key of the oldest brother of a given key.
   *
   * The oldest brother is defined as the first octant of a family of 2^dim octants.
   *
   * \param[in] key a given orchard key
   *
   * \return orchard key of the oldest brother.
   */
  KOKKOS_INLINE_FUNCTION
  static uint64_t
  oldest_brother(uint64_t key)
  {
    // oldest brother must have coordinates that are divisible by
    // 2*octantLength we just need to zero-out the bits of the remainder

    const auto lev = level(key);

    // 1. remove all the bits
    uint64_t key_new = key >> (DIM * (NUM_LEVELS - lev) + LEVEL_BITWIDTH + OUTSIDE_BITWIDTH);

    key_new = (key_new << (DIM * (NUM_LEVELS - lev) + LEVEL_BITWIDTH + OUTSIDE_BITWIDTH));

    set_level(key_new, level(key));
    set_outside_status(key_new, outside_status(key));

    return key_new;

  } // oldest_brother

  /**
   * Get the orchard key of the oldest brother of a given key (old version).
   *
   * The oldest brother is defined as the first octant of a family of 2^dim octants.
   *
   * \param[in] key is a given orchard key
   *
   * \return orchard key of the oldest brother.
   */
  KOKKOS_INLINE_FUNCTION
  static uint64_t
  oldest_brother_old(uint64_t key)
  {

    // octant logical length
    const auto length = octantLength(key);

    // twice octant size minus 1
    const uint32_t mask = (1 << (NUM_LEVELS - level(key))) - 1;

    // input key octant coordinates
    auto x = get_octant_coord<IX>(key);
    auto y = get_octant_coord<IY>(key);
    auto z = get_octant_coord<IZ>(key);

    // get oldest brother octant input key
    x = (x & mask) == 0 ? x : x - length;
    y = (y & mask) == 0 ? y : y - length;
    z = (z & mask) == 0 ? z : z - length;

    // oldest brother has the same level, and belongs to the same tree
    const auto oct_level = level(key);
    const auto tree_id = morton_tree(key);

    return encode_orchard(tree_id, { x, y, z }, oct_level, outside_status(key));

  } // oldest_brother_old

  /**
   * Is the first octant of a family of 2^dim octants ?
   *
   * \param[in] orchard_key
   */
  KOKKOS_INLINE_FUNCTION
  static bool
  isFirstOctantOfFamily(uint64_t orchard_key)
  {

    // // twice octant size minus 1
    // const uint32_t mask = (1 << (NUM_LEVELS - level(orchard_key))) - 1;

    // // check if x,y,z are integral multiple of mask+1 (= 2x octant length)
    // return (((get_octant_coord<IX>(orchard_key) & mask) == 0) and
    //         ((get_octant_coord<IY>(orchard_key) & mask) == 0) and
    //         ((get_octant_coord<IZ>(orchard_key) & mask) == 0));

    return orchard_key == oldest_brother(orchard_key);

  } // isFirstOctantOfFamily

  /**
   * Get the orchard key of the father of a given key.
   *
   * The father is defined as the octant at level l-1 (coarse) with the same coordinate as the
   * oldest brother.
   *
   * \param[in] key a given orchard key
   *
   * \return orchard key of the father
   */
  KOKKOS_INLINE_FUNCTION
  static uint64_t
  father(uint64_t key)
  {

    // octant logical length
    const auto length = octantLength(key);

    // twice octant size minus 1
    const uint32_t mask = (1 << (NUM_LEVELS - level(key))) - 1;

    // input key octant coordinates
    auto x = get_octant_coord<IX>(key);
    auto y = get_octant_coord<IY>(key);
    auto z = get_octant_coord<IZ>(key);

    // get oldest brother octant input key
    x = (x & mask) == 0 ? x : x - length;
    y = (y & mask) == 0 ? y : y - length;
    z = (z & mask) == 0 ? z : z - length;

    // father octant is at coarse level, and belongs to the same tree
    const auto oct_level = level(key) - 1;
    const auto tree_id = morton_tree(key);

    return encode_orchard(tree_id, { x, y, z }, oct_level, outside_status(key));

  } // father

  /**
   * Compute child id of given key with respect to its father octant.
   *
   * The father is defined as the octant at level l-1 (coarse) with the same coordinate as the
   * oldest brother.
   *
   * \param[in] key a given orchard key
   *
   * \return child id
   */
  KOKKOS_INLINE_FUNCTION
  static uint8_t
  child_id(uint64_t key)
  {
    uint8_t id = 0;

    // twice octant size minus 1
    const uint32_t mask = (1 << (NUM_LEVELS - level(key))) - 1;

    // input key octant coordinates
    auto x = get_octant_coord<IX>(key);
    auto y = get_octant_coord<IY>(key);
    auto z = get_octant_coord<IZ>(key);

    // get oldest brother octant input key
    id = (x & mask) == 0 ? id : id + 1;
    id = (y & mask) == 0 ? id : id + 2;
    id = (z & mask) == 0 ? id : id + 4;

    return id;

  } // child_id

  /**
   * Get the orchard key of the eldest child of a given key.
   *
   * The eldest child is defined as the octant at level l+1 (refined) with the same coordinate as
   * the father.
   *
   * \param[in] key a given orchard key
   *
   * \return orchard key of the eldest child
   */
  KOKKOS_INLINE_FUNCTION
  static uint64_t
  eldest_child(uint64_t key)
  {

    uint64_t child_key = key;

    set_level(child_key, level(key) + 1);

    return child_key;

  } // eldest_child

  /**
   * Get the orchard keys of all the children of a given key.
   *
   * \param[in] key a given orchard key
   *
   * \return array of orchard keys of all the children
   */
  KOKKOS_INLINE_FUNCTION
  static auto
  all_children(uint64_t key) -> Kokkos::Array<uint64_t, NB_CHILDREN>
  {

    Kokkos::Array<uint64_t, NB_CHILDREN> res;

    // child octant logical length
    const auto length = octantLength(key) >> 1;

    // children belong to the same tree
    const auto tree_id = morton_tree(key);

    const auto oct_level = level(key) + 1;

    // input key octant coordinates
    auto x = get_octant_coord<IX>(key);
    auto y = get_octant_coord<IY>(key);
    auto z = get_octant_coord<IZ>(key);

    // eldest child
    // res[0] = eldest_child(key);

    // sweep children in Morton order
    for (uint32_t i = 0; i < NB_CHILDREN; ++i)
    {
      // uint8_t ix = i & 0x1;
      // uint8_t iy = (i & 0x2) >> 1;
      // uint8_t iz = (i & 0x4) >> 2;
      res[i] = encode_orchard(
        tree_id,
        { x + (i & 0x1) * length, y + ((i & 0x2) >> 1) * length, z + ((i & 0x4) >> 2) * length },
        oct_level,
        outside_status(key));
    }

    return res;

  } // all_children

  /**
   * Get the orchard key of the nth child of a given key.
   * Remember the are NB_CHILDREN child octant, enumerated in Morton order.
   *
   * \param[in] key a given orchard key
   *
   * \return orchard key of the nth child
   */
  KOKKOS_INLINE_FUNCTION
  static uint64_t
  child(uint64_t key, uint8_t iChild)
  {

    KOKKOS_ASSERT(iChild < NB_CHILDREN && "iChild is not less than NB_CHILDREN");

    // child octant logical length
    const auto length = octantLength(key) >> 1;

    // children belong to the same tree
    const auto tree_id = morton_tree(key);

    const auto oct_level = level(key) + 1;

    // input key octant coordinates
    auto x = get_octant_coord<IX>(key);
    auto y = get_octant_coord<IY>(key);
    auto z = get_octant_coord<IZ>(key);

    return encode_orchard(tree_id,
                          { x + (iChild & 0x1) * length,
                            y + ((iChild & 0x2) >> 1) * length,
                            z + ((iChild & 0x4) >> 2) * length },
                          oct_level,
                          outside_status(key));

  } // child

  /**
   *
   * For a given face (Face::XMIN, etc...), return the smallest child of a finer neighbor that
   * "touches" current quadrant through that given face.
   *
   * In the drawing below (2D), the numbers are child ids we are talking about.
   *
   * This function is used when, under the assumption neighbor is at finer level, that actually
   * there is a neighbor at finer level by probing the hashmap.
   *
   *          --------
   *         |   |   |
   *         |___|___|
   *         |   |   |
   *  ___ ___|_0_|___|________
   * |   |   |       |   |   |
   * |___|___|       |___|___|
   * |   |   |       |   |   |
   * |___|_1_|_______|_0_|___|
   *         | 2 |   |
   *         |___|___|
   *         |   |   |
   *         |___|___|
   */
  KOKKOS_INLINE_FUNCTION
  static uint8_t
  get_face_neighbor_smallest_child_id(Face::face_t face)
  {
    // if (face == Face::XMIN)
    //   return 1;
    // else if (face == Face::XMAX)
    //   return 0;
    // else if (face == Face::YMIN)
    //   return 2;
    // else if (face == Face::YMAX)
    //   return 0;
    // else if (face == Face::ZMIN)
    //   return 4;
    // else if (face == Face::ZMAX)
    //   return 0;

    // if face is a right face (odd number) then return 0
    // else 2**(face/2)
    return (face & 0x1) == 1 ? 0 : (1 << (face / 2));

  } // get_face_neighbor_smallest_child_id

  /**
   *
   * For a given edge/corner (identified by intersection of two faces),
   * return the smallest child of a finer neighbor that
   * "touches" current quadrant through that given edge/corner.
   *
   * In the drawing below, the numbers are child ids we are talking about.
   *
   * This function is used when, under the assumption neighbor is at finer level, that actually
   * there is a neighbor at finer level by probing the hashmap.
   *
   *
   *  ________        ________
   * |   |   |       |   |   |
   * |___|___|       |___|___|
   * |   | 1 |       | 0 |   |
   * |___|___|_______|___|___|
   *         |       |
   *         |       |
   *         |       |
   *  _______|_______|________
   * |   | 3 |       | 2 |   |
   * |___|___|       |___|___|
   * |   |   |       |   |   |
   * |___|___|       |___|___|
   */
  KOKKOS_INLINE_FUNCTION
  static uint8_t
  get_edge_neighbor_smallest_child_id(Face::face_t face0, Face::face_t face1)
  {
    const auto dir0 = face0 / 2;
    const auto dir1 = face1 / 2;
    if (dir0 == 0 and dir1 == 1)
    {
      return ((~face0) & 0x1) + (((~face1) & 0x1) << 1);
    }
    else if (dir0 == 1 and dir1 == 2)
    {
      return (((~face0) & 0x1) + (((~face1) & 0x1) << 1)) << 1;
    }
    else if (dir0 == 0 and dir1 == 2)
    {
      return (((~face0) & 0x1) + (((~face1) & 0x1) << 2));
    }
    else
    {
      // we should not be here
      KOKKOS_ASSERT(false && "There is obviously something wrong in mesh connectivity....");
      return 0;
    }

  } // get_edge_neighbor_smallest_child_id

  KOKKOS_INLINE_FUNCTION
  static uint8_t
  get_corner_neighbor_smallest_child_id(Face::face_t face0, Face::face_t face1, Face::face_t face2)
  {
    return ((~face0) & 0x1) + (((~face1) & 0x1) << 1) + (((~face2) & 0x1) << 2);
  }

  /**
   * Compute the family id of a given orchard key.
   *
   * Family id is a number in [0, (2**dim)-1] is the local Morton index in a family of octant
   * (neighbors at the same level).
   *
   * \param[in] key a given orchard key
   *
   * \return family
   */
  KOKKOS_INLINE_FUNCTION static uint32_t
  family_id(uint64_t key)
  {

    return static_cast<uint32_t>(
      (key ^ oldest_brother(key)) >>
      (DIM * (NUM_LEVELS - level(key) - 1) + LEVEL_BITWIDTH + OUTSIDE_BITWIDTH));

  } // family_id

  /**
   * Returns true if current octant (identified by an orchard key) touches the border of current
   * tree through given face.
   *
   * \param[in] key is the orchard key of an octant to probe
   * \param[in] face identifies the face we query about being at domain border
   */
  KOKKOS_INLINE_FUNCTION
  static bool
  is_at_tree_border(uint64_t orchard_key, Face::face_t face)
  {

    auto x = get_octant_coord<IX>(orchard_key);
    auto y = get_octant_coord<IY>(orchard_key);
    auto z = get_octant_coord<IZ>(orchard_key);

    auto lev = level(orchard_key);

    // get octant logical size minus 1
    const uint32_t OCTANT_SIZE_M1 = (1 << (NUM_LEVELS - lev - 1)) - 1;

    if (face == Face::XMIN and x == 0)
    {
      return true;
    }

    if (face == Face::YMIN and y == 0)
    {
      return true;
    }

    if (face == Face::ZMIN and z == 0)
    {
      return true;
    }

    if (face == Face::XMAX and (((x ^ OCTANT_SIZE_M1) & TREE_SIZE_MASK) == TREE_SIZE_MASK))
    {
      return true;
    }

    if (face == Face::YMAX and (((y ^ OCTANT_SIZE_M1) & TREE_SIZE_MASK) == TREE_SIZE_MASK))
    {
      return true;
    }

    if (face == Face::ZMAX and (((z ^ OCTANT_SIZE_M1) & TREE_SIZE_MASK) == TREE_SIZE_MASK))
    {
      return true;
    }

    // returning default value
    return false;

  } // is_at_tree_border

  /**
   * Returns true if current octant (identified by an orchard key) touches the border of current
   * tree through given face.
   *
   * \param[in] key is the orchard key of an octant to probe
   * \param[in] face identifies the face we query about being at domain border
   */
  KOKKOS_INLINE_FUNCTION
  static bool
  is_at_any_tree_border(uint64_t orchard_key)
  {

    // clang-format off
    return is_at_tree_border(orchard_key, Face::XMIN) or
           is_at_tree_border(orchard_key, Face::XMAX) or
           is_at_tree_border(orchard_key, Face::YMIN) or
           is_at_tree_border(orchard_key, Face::YMAX) or
           is_at_tree_border(orchard_key, Face::ZMIN) or
           is_at_tree_border(orchard_key, Face::ZMAX);
    // clang-format on

  } // is_at_any_tree_border

  /**
   * Returns true if current octant (identified by an orchard key) touches the outer domain boundary
   * through given face.
   *
   * \param[in] key is the orchard key of an octant to probe
   * \param[in] face identifies the face we query about being at domain border
   * \param[in] brick_sizes is an array of sizes of the p4est brick connectivity
   *
   */
  KOKKOS_INLINE_FUNCTION
  static bool
  is_at_domain_border(uint64_t key, Face::face_t face, brick_size_t<DIM> const & brick_sizes)
  {

    auto tree_x = get_tree_coord<IX>(key);
    auto tree_y = get_tree_coord<IY>(key);
    auto tree_z = get_tree_coord<IZ>(key);

    if (face == Face::XMIN and tree_x == 0 and is_at_tree_border(key, face))
      return true;

    if (face == Face::YMIN and tree_y == 0 and is_at_tree_border(key, face))
      return true;

    if (face == Face::ZMIN and tree_z == 0 and is_at_tree_border(key, face))
      return true;

    if (face == Face::XMAX and (tree_x == (brick_sizes[IX] - 1)) and is_at_tree_border(key, face))
      return true;

    if (face == Face::YMAX and (tree_y == (brick_sizes[IY] - 1)) and is_at_tree_border(key, face))
      return true;

    if (face == Face::ZMAX and (tree_z == (brick_sizes[IZ] - 1)) and is_at_tree_border(key, face))
      return true;

    // returning default value
    return false;

  } // is_at_domain_border

  /**
   * Returns true if current octant (identified by an orchard key) touches any outer domain
   * boundary.
   *
   * \param[in] key is the orchard key of an octant to probe
   * \param[in] brick_sizes is an array of sizes of the p4est brick connectivity
   *
   */
  KOKKOS_INLINE_FUNCTION
  static bool
  is_at_any_domain_border(uint64_t key, brick_size_t<DIM> const & brick_sizes)
  {

    auto tree_x = get_tree_coord<IX>(key);
    auto tree_y = get_tree_coord<IY>(key);
    auto tree_z = get_tree_coord<IZ>(key);

    bool result = false;

    if ((tree_x == 0 and is_at_tree_border(key, Face::XMIN)) or
        (tree_y == 0 and is_at_tree_border(key, Face::YMIN)) or
        (tree_z == 0 and is_at_tree_border(key, Face::ZMIN)) or
        ((tree_x == (brick_sizes[IX] - 1)) and is_at_tree_border(key, Face::XMAX)) or
        ((tree_y == (brick_sizes[IY] - 1)) and is_at_tree_border(key, Face::YMAX)) or
        ((tree_z == (brick_sizes[IZ] - 1)) and is_at_tree_border(key, Face::ZMAX)))
      result = true;

    return result;

  } // is_at_any_domain_border

  /**
   * Returns true if current octant (identified by an orchard key) touches the border of current
   * tree through two given face: face1 and face2.
   *
   * Being at edge means we touch two faces border.
   *
   * \param[in] key is the orchard key of an octant to probe
   * \param[in] face1 identifies the first face
   * \param[in] face2 identifies the second face
   */
  KOKKOS_INLINE_FUNCTION
  static bool
  is_at_tree_edge(uint64_t orchard_key, Face::face_t face1, Face::face_t face2)
  {

    return is_at_tree_border(orchard_key, face1) and is_at_tree_border(orchard_key, face2);

  } // is_at_tree_edge

  /**
   * Returns true if current octant (identified by an orchard key) touches the border of current
   * tree through any edge.
   */
  KOKKOS_INLINE_FUNCTION
  static bool
  is_at_any_tree_edge(uint64_t orchard_key)
  {

    return is_at_tree_edge(orchard_key, Face::XMIN, Face::YMIN) or
           is_at_tree_edge(orchard_key, Face::XMIN, Face::YMAX) or
           is_at_tree_edge(orchard_key, Face::XMAX, Face::YMIN) or
           is_at_tree_edge(orchard_key, Face::XMAX, Face::YMAX) or
           is_at_tree_edge(orchard_key, Face::YMIN, Face::ZMIN) or
           is_at_tree_edge(orchard_key, Face::YMIN, Face::ZMAX) or
           is_at_tree_edge(orchard_key, Face::YMAX, Face::ZMIN) or
           is_at_tree_edge(orchard_key, Face::YMAX, Face::ZMAX) or
           is_at_tree_edge(orchard_key, Face::ZMIN, Face::XMIN) or
           is_at_tree_edge(orchard_key, Face::ZMIN, Face::XMAX) or
           is_at_tree_edge(orchard_key, Face::ZMAX, Face::XMIN) or
           is_at_tree_edge(orchard_key, Face::ZMAX, Face::XMAX);

  } // is_at_any_tree_edge

  /**
   * Returns true if current octant (identified by an orchard key) touches the border of current
   * tree through three given face: face1, face2 and face3.
   *
   * Being at corner means we touch three faces border.
   *
   * \param[in] key is the orchard key of an octant to probe
   * \param[in] face1 identifies the first face
   * \param[in] face2 identifies the second face
   * \param[in] face3 identifies the third face
   */
  KOKKOS_INLINE_FUNCTION
  static bool
  is_at_tree_corner(uint64_t     orchard_key,
                    Face::face_t face1,
                    Face::face_t face2,
                    Face::face_t face3)
  {

    return is_at_tree_border(orchard_key, face1) and is_at_tree_border(orchard_key, face2) and
           is_at_tree_border(orchard_key, face3);

  } // is_at_tree_corner

  /**
   * Returns true if current octant (identified by an orchard key) touches the border of current
   * tree through a corner.
   *
   * \param[in] key is the orchard key of an octant to probe
   */
  KOKKOS_INLINE_FUNCTION
  static bool
  is_at_any_tree_corner(uint64_t orchard_key)
  {

    return is_at_tree_corner(orchard_key, Face::XMIN, Face::YMIN, Face::ZMIN) or
           is_at_tree_corner(orchard_key, Face::XMIN, Face::YMIN, Face::ZMAX) or
           is_at_tree_corner(orchard_key, Face::XMIN, Face::YMAX, Face::ZMIN) or
           is_at_tree_corner(orchard_key, Face::XMIN, Face::YMAX, Face::ZMAX) or
           is_at_tree_corner(orchard_key, Face::XMAX, Face::YMIN, Face::ZMIN) or
           is_at_tree_corner(orchard_key, Face::XMAX, Face::YMIN, Face::ZMAX) or
           is_at_tree_corner(orchard_key, Face::XMAX, Face::YMAX, Face::ZMIN) or
           is_at_tree_corner(orchard_key, Face::XMAX, Face::YMAX, Face::ZMAX);

  } // is_at_any_tree_corner

  /**
   * Return a DIM-dimensional integer vector which direction is the outside normal vector.
   *
   * - this vector is only non-zero for key that touches the external border.
   * - for keys that represent a block that is at corner of domain, (e.g. lower left corner, normal
   *   is {-1, -1}), the norm of the vector is not one.
   * - for keys that represent a block that touches a face but not a corner, the norm of this vector
   *   is one.
   *
   * \note in early version, if brick connectivity was not periodic, normal vector was null, but in
   * current version normal vector at border is always not null.
   *
   * \param[in] key is the orchard key of an octant to probe
   * \param[in] brick_sizes is an array of sizes of the p4est brick connectivity
   * \param[in] is_brick_periodic is array of bool (one per direction) to specify if p4est
   *            connectivity is periodic
   */
  KOKKOS_INLINE_FUNCTION
  static Kokkos::Array<int8_t, DIM>
  get_outside_normal(uint64_t                                          orchard_key,
                     brick_size_t<DIM> const &                         brick_sizes,
                     [[maybe_unused]] Kokkos::Array<bool, DIM> const & is_brick_periodic)
  {

    Kokkos::Array<int8_t, DIM> normal{ 0, 0, 0 };

    // if (!is_brick_periodic[IX])
    {
      if (is_at_domain_border(orchard_key, Face::XMIN, brick_sizes))
        normal[IX] = -1;
      if (is_at_domain_border(orchard_key, Face::XMAX, brick_sizes))
        normal[IX] = 1;
    }

    // if (!is_brick_periodic[IY])
    {
      if (is_at_domain_border(orchard_key, Face::YMIN, brick_sizes))
        normal[IY] = -1;
      if (is_at_domain_border(orchard_key, Face::YMAX, brick_sizes))
        normal[IY] = 1;
    }

    // if (!is_brick_periodic[IZ])
    {
      if (is_at_domain_border(orchard_key, Face::ZMIN, brick_sizes))
        normal[IZ] = -1;
      if (is_at_domain_border(orchard_key, Face::ZMAX, brick_sizes))
        normal[IZ] = 1;
    }

    return normal;

  } // get_outside_normal

  /**
   * Get orchard key of a neighbor across a corner
   * a given input octant orchard key, assuming neighbor is at same level.
   *
   * A corner is identified by (intersection of) 3 orthogonal faces
   *
   * \param[in] key is the orchard key of current octant
   * \param[in] face_x is a faceId (Face::XMIN, Face::XMAX, ....)
   * \param[in] face_y is a faceId (Face::YMIN, Face::YMAX, ....)
   * \param[in] face_z is a faceId (Face::ZMIN, Face::ZMAX, ....)
   * \param[in] brick_sizes is a array containing number of trees per dimension for each dimension
   *
   * \return neighbor across a corner orchard key
   *
   * \deprecated { should prefer using get_neighbor_key_same_level }
   */
  KOKKOS_INLINE_FUNCTION
  static uint64_t
  get_corner_neighbor_key(uint64_t                  key,
                          Face::face_t              face_x,
                          Face::face_t              face_y,
                          Face::face_t              face_z,
                          brick_size_t<DIM> const & brick_sizes)
  {

    const auto lev = level(key);

    const auto x = get_octant_coord<IX>(key);
    const auto y = get_octant_coord<IY>(key);
    const auto z = get_octant_coord<IZ>(key);

    const auto tree = morton_tree(key);
    const auto tree_x = get_tree_coord<IX>(key);
    const auto tree_y = get_tree_coord<IY>(key);
    const auto tree_z = get_tree_coord<IZ>(key);

    auto tree_neigh = tree;

    // neighbor tree coordinates
    uint16_t tree_neigh_x = tree_x;
    uint16_t tree_neigh_y = tree_y;
    uint16_t tree_neigh_z = tree_z;

    // neighbor octant coordinates
    uint32_t x_neigh = x;
    uint32_t y_neigh = y;
    uint32_t z_neigh = z;

    const auto length = octantLength(key);

    // face_x
    if (face_x == Face::XMIN)
    {
      x_neigh = (x == 0) ? last_offset(lev) : x - length;

      // is neighbor octant in a different tree ?
      if (x == 0)
      {
        tree_neigh_x = tree_x == 0 ? brick_sizes[IX] - 1 : tree_x - 1;
      }
    }
    else if (face_x == Face::XMAX)
    {
      x_neigh = (x == last_offset(lev)) ? 0 : x + length;

      // is neighbor octant in a different tree ?
      if (x == last_offset(lev))
      {
        tree_neigh_x = tree_x == brick_sizes[IX] - 1 ? 0 : tree_x + 1;
      }
    }

    // face_y
    if (face_y == Face::YMIN)
    {
      y_neigh = (y == 0) ? last_offset(lev) : y - length;

      // is neighbor octant in a different tree ?
      if (y == 0)
      {
        tree_neigh_y = tree_y == 0 ? brick_sizes[IY] - 1 : tree_y - 1;
      }
    }
    else if (face_y == Face::YMAX)
    {
      y_neigh = (y == last_offset(lev)) ? 0 : y + length;

      // is neighbor octant in a different tree ?
      if (y == last_offset(lev))
      {
        tree_neigh_y = tree_y == brick_sizes[IY] - 1 ? 0 : tree_y + 1;
      }
    }

    // face_z
    if (face_z == Face::ZMIN)
    {
      z_neigh = (z == 0) ? last_offset(lev) : z - length;

      // is neighbor octant in a different tree ?
      if (z == 0)
      {
        tree_neigh_z = tree_z == 0 ? brick_sizes[IZ] - 1 : tree_z - 1;
      }
    }
    else if (face_z == Face::ZMAX)
    {
      z_neigh = (z == last_offset(lev)) ? 0 : z + length;

      // is neighbor octant in a different tree ?
      if (z == last_offset(lev))
      {
        tree_neigh_z = tree_z == brick_sizes[IZ] - 1 ? 0 : tree_z + 1;
      }
    }

    tree_neigh = encode_morton_tree(tree_neigh_x, tree_neigh_y, tree_neigh_z);

    return encode_orchard(tree_neigh, { x_neigh, y_neigh, z_neigh }, lev, outside_status(key));

  } // get_corner_neighbor_key

  /**
   * Get orchard key of a face neighbor of
   * a given input octant orchard key, assuming neighbor is at same level.
   *
   * \param[in] key is the orchard key of current octant
   * \param[in] face is faceId (Face::XMIN, Face::XMAX, ....)
   * \param[in] brick_sizes is a array containing number of trees per dimension for each dimension
   * \param[in] level_neigh neighbor level
   *
   * Current octant (identified by key) and neighbor octant must have at most 1 level difference.
   *
   * \return neighbor orchard key
   */
  KOKKOS_INLINE_FUNCTION
  static uint64_t
  get_face_neighbor_key(uint64_t key, Face::face_t face, brick_size_t<DIM> const & brick_sizes)
  {

    return get_corner_neighbor_key(key, face, face, face, brick_sizes);

  } // get_face_neighbor_key

  /**
   * Get orchard key of a neighbor of a given octant, in a given direction at same level.
   *
   * \param[in] key is the orchard key of current octant
   * \param[in] direction is a vector pointing to one of the 3x3-1 neighbors, component are -1,0 or
   * 1
   * \param[in] brick_sizes is an array containing number of trees per dimension for each dimension
   * \param[in] is_brick_periodic is an array of boolean (one value per direction) stating if mesh
   * is periodic
   *
   * \return neighbor orchard key
   */
  KOKKOS_INLINE_FUNCTION
  static uint64_t
  get_neighbor_key_same_level(uint64_t                           key,
                              Kokkos::Array<int8_t, DIM> const & direction,
                              brick_size_t<DIM> const &          brick_sizes,
                              Kokkos::Array<bool, DIM> const &   is_brick_periodic)
  {

    const auto lev = level(key);
    const auto length = octantLength(key);

    const auto xy = get_octant_coords(key);

    const auto tree = morton_tree(key);
    const auto tree_xy = get_tree_coords(key);

    auto tree_neigh = tree;

    // neighbor tree coordinates
    auto tree_neigh_xy = tree_xy;

    // neighbor octant coordinates
    auto xy_neigh = xy;

    auto outside = outside_status(key);
    auto outside_neigh = outside;

    for (int dir = 0; dir < DIM; ++dir)
    {

      if (direction[dir] == -1)
      {
        xy_neigh[dir] = (xy[dir] == 0) ? last_offset(lev) : xy[dir] - length;

        // is neighbor octant in a different tree ?
        if (xy[dir] == 0)
        {
          tree_neigh_xy[dir] = tree_xy[dir] == 0 ? brick_sizes[dir] - 1 : tree_xy[dir] - 1;
        }

        // if mesh not periodic, we need to set outside status bits
        if (!is_brick_periodic[dir])
        {
          if (xy[dir] == 0 and tree_xy[dir] == 0)
          {
            outside_neigh = outside_neigh ^ (ONE_U << dir);
          }
        }
      }
      else if (direction[dir] == 1)
      {
        xy_neigh[dir] = (xy[dir] == last_offset(lev)) ? 0 : xy[dir] + length;

        // is neighbor octant in a different tree ?
        if (xy[dir] == last_offset(lev))
        {
          tree_neigh_xy[dir] = tree_xy[dir] == brick_sizes[dir] - 1 ? 0 : tree_xy[dir] + 1;
        }

        // if mesh not periodic, we need to set outside status bits
        if (!is_brick_periodic[dir])
        {
          if (xy[dir] == last_offset(lev) and tree_xy[dir] == brick_sizes[dir] - 1)
          {
            outside_neigh = outside_neigh ^ (ONE_U << dir);
          }
        }
      }

    } // end for dir

    tree_neigh = encode_morton_tree(tree_neigh_xy);

    return encode_orchard(tree_neigh, xy_neigh, lev, outside_neigh);

  } // get_neighbor_key_same_level

  KOKKOS_INLINE_FUNCTION
  static void
  face_to_displacement(Kokkos::Array<int8_t, DIM> & displacement, Face::face_t face)
  {

    if (face == Face::XMIN)
      displacement[IX] = -1;
    else if (face == Face::XMAX)
      displacement[IX] = 1;
    else if (face == Face::YMIN)
      displacement[IY] = -1;
    else if (face == Face::YMAX)
      displacement[IY] = 1;
    else if (face == Face::ZMIN)
      displacement[IZ] = -1;
    else if (face == Face::ZMAX)
      displacement[IZ] = 1;

  } // face_to_displacement

  /**
   * return a displacement vector from a face id.
   */
  KOKKOS_INLINE_FUNCTION
  static Kokkos::Array<int8_t, DIM>
  face_to_displacement(Face::face_t face)
  {

    Kokkos::Array<int8_t, DIM> displacement{ 0, 0, 0 };

    face_to_displacement(displacement, face);

    return displacement;

  } // face_to_displacement

  /**
   * return a displacement vector from a pair of face id.
   */
  KOKKOS_INLINE_FUNCTION
  static Kokkos::Array<int8_t, DIM>
  face_to_displacement(Face::face_t face0, Face::face_t face1)
  {

    Kokkos::Array<int8_t, DIM> displacement{ 0, 0, 0 };

    face_to_displacement(displacement, face0);
    face_to_displacement(displacement, face1);

    return displacement;

  } // face_to_displacement

  /**
   * return a displacement vector from a pair of face id.
   */
  KOKKOS_INLINE_FUNCTION
  static Kokkos::Array<int8_t, DIM>
  face_to_displacement(Face::face_t face0, Face::face_t face1, Face::face_t face2)
  {

    Kokkos::Array<int8_t, DIM> displacement{ 0, 0, 0 };

    face_to_displacement(displacement, face0);
    face_to_displacement(displacement, face1);
    face_to_displacement(displacement, face2);

    return displacement;

  } // face_to_displacement

  /**
   * decode and print orchard key (for debug).
   */
  KOKKOS_INLINE_FUNCTION
  static void
  print(uint64_t key, const char * str = "")
  {
    auto tree_coord = get_tree_coords(key);
    auto oct_coord = get_octant_coords(key);

    printf("[%s] key: %ld | tree_morton : %d | tree_coord : %d %d %d | octant coord: %d %d %d | "
           "outside: %d%d%d | level: %d\n",
           str,
           key,
           morton_tree(key),
           tree_coord[IX],
           tree_coord[IY],
           tree_coord[IZ],
           oct_coord[IX],
           oct_coord[IY],
           oct_coord[IZ],
           is_touching_face_X(key),
           is_touching_face_Y(key),
           is_touching_face_Z(key),
           level(key));
  } // print

}; // struct orchard_key_t<3>

KALYPSSO_DISABLE_CONVERSION_WARNINGS_POP()
KALYPSSO_DISABLE_NVCC_WARNINGS_POP()
