// SPDX-FileCopyrightText: 2025 kalypsso-core authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

KALYPSSO_DISABLE_CONVERSION_WARNINGS_PUSH()
KALYPSSO_DISABLE_NVCC_WARNINGS_PUSH()

/**
 * \file orchard_key_impl_2d.h
 */
// ====================================================================
// ====================================================================
// ====================================================================
template <>
struct orchard_key_t<2> : public BitFieldInteger<key_t>
{
  orchard_key_t(const orchard_key_t &) = default;

  orchard_key_t(orchard_key_t &&) = default;

  orchard_key_t &
  operator=(const orchard_key_t &) = default;

  orchard_key_t &
  operator=(orchard_key_t &&) = default;

  static constexpr uint8_t ONE_U = 1;

  /** dimension */
  static constexpr uint8_t DIM = 2;

  /** number of bits used to encode outside status */
  static constexpr uint8_t OUTSIDE_BITWIDTH = 2;

  /** number of bits used to encode level */
  static constexpr uint8_t LEVEL_BITWIDTH = 6;

  /** number of bits used to encode morton octant */
  static constexpr uint8_t MORTON_OCTANT_BITWIDTH = 44;

  /** number of bits used to encode morton tree */
  static constexpr uint8_t MORTON_TREE_BITWIDTH = 12;

  /** bit offset to the first bit where outside status bits start */
  static constexpr uint8_t OUTSIDE_OFFSET = 0;

  /** bit offset to the first bit where level bits start : 0 + 2 = 2 */
  static constexpr uint8_t LEVEL_OFFSET = OUTSIDE_OFFSET + OUTSIDE_BITWIDTH;

  /** bit offset to the first bit where the morton octant bits start : 2 + 6 = 8 */
  static constexpr uint8_t MORTON_OCTANT_OFFSET = LEVEL_OFFSET + LEVEL_BITWIDTH;

  /** bit offset to the first bit where tree bits start : 8 + 44 = 52 */
  static constexpr uint8_t MORTON_TREE_OFFSET = MORTON_OCTANT_OFFSET + MORTON_OCTANT_BITWIDTH;

  /** bits for outside status */
  static constexpr uint64_t OUTSIDE_MASK = 0x0000000000000003;

  /** bit mask to extract LEVEL (6 bits right after outside status) */
  static constexpr uint64_t LEVEL_MASK = 0x00000000000000FC;

  /** bit mask to extract LEVEL (the middle 44 bits) */
  static constexpr uint64_t MORTON_OCTANT_MASK = 0x000FFFFFFFFFFF00;

  /** bit mask to extract MORTON_TREE (the upper 12 bits) */
  static constexpr uint64_t MORTON_TREE_MASK = 0xFFF0000000000000;

  /** level is in range [0, MAX_LEVEL] */
  static constexpr uint16_t MAX_LEVEL = MORTON_OCTANT_BITWIDTH / 2;

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

  /** number of neighbors across an edge at fine level (no edge in 2d) */
  static constexpr uint32_t NB_EDGE_NEIGHBORS_FINE = 0;

  /** maximum number of neighbors */
  static constexpr uint32_t MAX_NUM_NEIGHBORS = NB_CHILDREN / 2;

  using BitFieldInteger::BitFieldInteger;
  DECLARE_BIT(is_touching_face_X, OUTSIDE_OFFSET)
  DECLARE_BIT(is_touching_face_Y, OUTSIDE_OFFSET + 1)
  DECLARE_CASTED_FIELD(outside_status, OUTSIDE_OFFSET, OUTSIDE_BITWIDTH, uint8_t)
  DECLARE_CASTED_FIELD(level, LEVEL_OFFSET, LEVEL_BITWIDTH, uint8_t)
  DECLARE_CASTED_FIELD(morton_octant, MORTON_OCTANT_OFFSET, MORTON_OCTANT_BITWIDTH, uint64_t)
  DECLARE_CASTED_FIELD(morton_tree, MORTON_TREE_OFFSET, MORTON_TREE_BITWIDTH, uint16_t)

  // ====================================================================
  //! get the outside status of a given key / quadrant
  KOKKOS_INLINE_FUNCTION static bool
  is_outside(uint64_t const & key)
  {

    return is_touching_face_X(key) or is_touching_face_Y(key);

  } // is_outside

  // ====================================================================
  //! return true only if key is outside and "touching" inside domain by face along given "dir"
  //! \note a corner outside is touching inside domain by 2 faces.
  KOKKOS_INLINE_FUNCTION static bool
  is_outside_dir(uint64_t const & key, Dir::dir_t dir)
  {

    return ((dir == Dir::X) and is_touching_face_X(key)) or
           ((dir == Dir::Y) and is_touching_face_Y(key));

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

  } // set_outside_bit

  // ====================================================================
  //! change outside status. By resetting these bits, key becomes an inside key
  KOKKOS_INLINE_FUNCTION static void
  reset_outside_bits(uint64_t & key)
  {
    set_is_touching_face_X(key, false);
    set_is_touching_face_Y(key, false);
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
    static_assert(direction == IX || direction == IY, "Only X and Y directions are possible in 2D");

    auto tree_coord = morton_tree(key);

    // select direction
    tree_coord = tree_coord >> direction;

    // only 7 bits are significant
    tree_coord &= 0x1555;

    tree_coord = (tree_coord ^ (tree_coord >> 1)) & 0x3333;
    tree_coord = (tree_coord ^ (tree_coord >> 2)) & 0x0f0f;
    tree_coord = (tree_coord ^ (tree_coord >> 4)) & 0x00ff;

    return static_cast<uint16_t>(tree_coord);

  } // get_tree_coord

  // ====================================================================
  KOKKOS_INLINE_FUNCTION static Kokkos::Array<uint16_t, DIM>
  get_tree_coords(uint64_t key)
  {
    // clang-format off
    return Kokkos::Array<uint16_t, DIM>{ get_tree_coord<IX>(key),
                                         get_tree_coord<IY>(key) };
    // clang-format on
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
    static_assert(direction == IX || direction == IY, "Only X and Y directions are possible in 2D");

    // extract bits corresponding to morton octant
    key = morton_octant(key);

    // select direction
    key = key >> direction;

    // only 22 bits are significant, so apply a 44 bits wide mask
    key &= 0x55555555555;

    // clang-format off
    key = (key ^ (key >>  1)) & 0x3333333333333333;
    key = (key ^ (key >>  2)) & 0x0f0f0f0f0f0f0f0f;
    key = (key ^ (key >>  4)) & 0x00ff00ff00ff00ff;
    key = (key ^ (key >>  8)) & 0x0000ffff0000ffff;
    key = (key ^ (key >> 16)) & 0x00000000ffffffff;
    // clang-format on

    return static_cast<uint32_t>(key);

  } // get_octant_coord

  // ====================================================================
  /**
   * get octant x,y or z coordinate from an orchard key.
   *
   * \param[in] key the orchard key
   */
  KOKKOS_INLINE_FUNCTION static Kokkos::Array<uint32_t, 2>
  get_octant_coords(uint64_t key)
  {
    // clang-format off
    return Kokkos::Array<uint32_t, DIM>{ get_octant_coord<IX>(key),
                                         get_octant_coord<IY>(key) };
    // clang-format on
  } // get_octant_coords

  // ====================================================================
  /**
   * Take an integer representing one of the tree coordinate, and modify its
   * binary representation as preliminary step before encoding morton key.
   *
   * \param[in] coord must be in range [0,63] (only 6 bits are significant)
   *
   * \return an integer with interleaved 0 in between each bits of input.
   */
  KOKKOS_INLINE_FUNCTION static uint16_t
  split_morton_tree(uint16_t coord)
  {
    // extract the last 6 bits
    uint16_t x = coord & 0x003F;

    x = (x | x << 4) & 0x0f0f;
    x = (x | x << 2) & 0x3333;
    x = (x | x << 1) & 0x5555;

    return x;

  } // split_morton_tree

  // ====================================================================
  /**
   * Take an integer representing one of the octant coordinate, and modify its
   * binary representation as preliminary step before encoding morton key.
   *
   * \param[in] coord must be in range [0,2^22-1] (only 22 bits are significant)
   *
   * the result is stored on a 64 bits integer because 2x22 bits = 44 bits (minimum)
   *
   * \return an integer with interleaved 0 in between each bits of input.
   */
  KOKKOS_INLINE_FUNCTION static uint64_t
  split_morton_octant(uint32_t coord)
  {
    // extract the last 22 bits
    uint64_t x = coord & 0x3FFFFF;

    // clang-format off
    x = (x | x << 16) &     0xffff0000ffff;
    x = (x | x <<  8) &   0xff00ff00ff00ff;
    x = (x | x <<  4) &  0xf0f0f0f0f0f0f0f;
    x = (x | x <<  2) & 0x3333333333333333;
    x = (x | x <<  1) & 0x5555555555555555;
    // clang-format on

    return x;

  } // split_morton_octant

  // ====================================================================
  /**
   * encode morton tree index from x,y coordinates
   * \param[in] treeCoord is an an array of tree coordinates in the p4est brick connectivity
   */
  KOKKOS_INLINE_FUNCTION static uint16_t
  encode_morton_tree(Kokkos::Array<uint16_t, 2> treeCoord)
  {
    uint16_t key = 0;
    key |= split_morton_tree(treeCoord[IX]) | split_morton_tree(treeCoord[IY]) << 1;

    return key;
  } // encode_morton_tree

  // ====================================================================
  /**
   * encode morton tree index from x,y coordinates
   * \param[in] tree_x is tree cartesian coordinate along X in the p4est brick connectivity
   * \param[in] tree_y is tree cartesian coordinate along Y in the p4est brick connectivity
   */
  KOKKOS_INLINE_FUNCTION static uint16_t
  encode_morton_tree(uint16_t tree_x, uint16_t tree_y)
  {
    uint16_t key = 0;
    key |= split_morton_tree(tree_x) | split_morton_tree(tree_y) << 1;

    return key;
  } // encode_morton_tree

  // ====================================================================
  /**
   * encode morton tree index from x,y coordinates
   *
   * \param[in,out] orchard_key
   * \param[in] treeCoord is an an array of tree coordinates in the p4est brick connectivity
   */
  KOKKOS_INLINE_FUNCTION static void
  encode_morton_tree(uint64_t & orchard_key, Kokkos::Array<uint16_t, 2> treeCoord)
  {
    uint16_t tree_key = 0;
    tree_key |= split_morton_tree(treeCoord[IX]) | split_morton_tree(treeCoord[IY]) << 1;

    set_morton_tree(orchard_key, tree_key);

  } // encode_morton_tree

  // ====================================================================
  /**
   * encode morton octant index from octant x,y coordinates
   * \param[in] octCoord must be in [0,2**22-1]^2
   */
  KOKKOS_INLINE_FUNCTION static void
  encode_morton_octant(uint64_t & orchard_key, Kokkos::Array<uint32_t, 2> octCoord)
  {
    uint64_t key = 0;
    key |= split_morton_octant(octCoord[IX]) | split_morton_octant(octCoord[IY]) << 1;

    set_morton_octant(orchard_key, key);

  } // encode_morton_octant

  // ====================================================================
  /**
   * encode orchard key from octant x,y coordinates, tree x,y coords and level
   * \param[in] octCoord must be in [0,2**22-1]^2
   *
   */
  KOKKOS_INLINE_FUNCTION static uint64_t
  encode_orchard(Kokkos::Array<uint16_t, 2> treeCoord,
                 Kokkos::Array<uint32_t, 2> octCoord,
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
   * encode orchard key from octant x,y coordinates, tree x,y coords and level
   * \param[in] octCoord must be in [0,2**22-1]^2
   *
   * key is always zero initialized, the outside status bits will be zero (i.e. inside quadrant).
   */
  KOKKOS_INLINE_FUNCTION static uint64_t
  encode_orchard(Kokkos::Array<uint16_t, 2> treeCoord,
                 Kokkos::Array<uint32_t, 2> octCoord,
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
   * encode orchard key from octant x,y coordinates, tree id and level
   * \param[in] octCoord must be in [0,2**22-1]^2
   *
   * key is always zero initialized, the outside bit will be zero.
   */
  KOKKOS_INLINE_FUNCTION static uint64_t
  encode_orchard(uint16_t                   treeid,
                 Kokkos::Array<uint32_t, 2> octCoord,
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

    // get oldest brother octant input key
    x = (x & mask) == 0 ? x : x - length;
    y = (y & mask) == 0 ? y : y - length;

    // oldest brother has the same level, and belongs to the same tree
    const auto oct_level = level(key);
    const auto tree_id = morton_tree(key);

    return encode_orchard(tree_id, { x, y }, oct_level, outside_status(key));

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
    //         ((get_octant_coord<IY>(orchard_key) & mask) == 0));

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

    // get oldest brother octant input key
    x = (x & mask) == 0 ? x : x - length;
    y = (y & mask) == 0 ? y : y - length;

    // father octant is at coarse level, and belongs to the same tree
    const auto oct_level = level(key) - 1;
    const auto tree_id = morton_tree(key);

    return encode_orchard(tree_id, { x, y }, oct_level, outside_status(key));

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

    // get oldest brother octant input key
    id = (x & mask) == 0 ? id : id + 1;
    id = (y & mask) == 0 ? id : id + 2;

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

    // eldest child
    // res[0] = eldest_child(key);

    // sweep children in Morton order
    for (uint32_t i = 0; i < NB_CHILDREN; ++i)
    {
      // uint8_t ix = i & 0x1;
      // uint8_t iy = (i & 0x2) >> 1;
      res[i] = encode_orchard(tree_id,
                              { x + (i & 0x1) * length, y + ((i & 0x2) >> 1) * length },
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

    return encode_orchard(tree_id,
                          { x + (iChild & 0x1) * length, y + ((iChild & 0x2) >> 1) * length },
                          oct_level,
                          outside_status(key));

  } // child

  /**
   *
   * For a given face (Face::XMIN, etc...), return the smallest child of a finer neighbor that
   * "touches" current quadrant through that given face.
   *
   * In the drawing below, the numbers are child ids we are talking about.
   *
   * This function is used when, under the assumption neighbor is at finer level, that actually
   * there is a neighbor at finer level by probing the hashmap.
   *
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
    KOKKOS_ASSERT(((face0 == Face::XMIN) or (face0 == Face::XMAX)) && "wrong face id");
    KOKKOS_ASSERT(((face1 == Face::YMIN) or (face1 == Face::YMAX)) && "wrong face id");

    return ((~face0) & 0x1) + (((~face1) & 0x1) << 1);

  } // get_edge_neighbor_smallest_child_id

  // ==============================================
  /**
   * For compatibility with 3d.
   *
   * only face0 and face1 are used.
   */
  KOKKOS_INLINE_FUNCTION
  static uint8_t
  get_corner_neighbor_smallest_child_id(Face::face_t                  face0,
                                        Face::face_t                  face1,
                                        [[maybe_unused]] Face::face_t face2)
  {
    KOKKOS_ASSERT(((face0 == Face::XMIN) or (face0 == Face::XMAX)) && "wrong face id");
    KOKKOS_ASSERT(((face1 == Face::YMIN) or (face1 == Face::YMAX)) && "wrong face id");

    return get_edge_neighbor_smallest_child_id(face0, face1);

  } // get_corner_neighbor_smallest_child_id

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
  KOKKOS_INLINE_FUNCTION
  static uint32_t
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

    if (face == Face::XMAX and (((x ^ OCTANT_SIZE_M1) & TREE_SIZE_MASK) == TREE_SIZE_MASK))
    {
      return true;
    }

    if (face == Face::YMAX and (((y ^ OCTANT_SIZE_M1) & TREE_SIZE_MASK) == TREE_SIZE_MASK))
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
           is_at_tree_border(orchard_key, Face::YMAX);
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

    if (face == Face::XMIN and tree_x == 0 and is_at_tree_border(key, face))
      return true;

    if (face == Face::YMIN and tree_y == 0 and is_at_tree_border(key, face))
      return true;

    if (face == Face::XMAX and (tree_x == (brick_sizes[IX] - 1)) and is_at_tree_border(key, face))
      return true;

    if (face == Face::YMAX and (tree_y == (brick_sizes[IY] - 1)) and is_at_tree_border(key, face))
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

    bool result = false;

    if ((tree_x == 0 and is_at_tree_border(key, Face::XMIN)) or
        (tree_y == 0 and is_at_tree_border(key, Face::YMIN)) or
        ((tree_x == (brick_sizes[IX] - 1)) and is_at_tree_border(key, Face::XMAX)) or
        ((tree_y == (brick_sizes[IY] - 1)) and is_at_tree_border(key, Face::YMAX)))
      result = true;

    return result;

  } // is_at_any_domain_border

  /**
   * Returns true if current octant (identified by an orchard key) touches the border of current
   * tree through given face1 and face2.
   *
   * Being at corner means we touch two faces border.
   *
   * \param[in] key is the orchard key of an octant to probe
   * \param[in] face1 identifies the first face
   * \param[in] face2 identifies the second face
   */
  KOKKOS_INLINE_FUNCTION
  static bool
  is_at_tree_corner(uint64_t orchard_key, Face::face_t face1, Face::face_t face2)
  {

    return is_at_tree_border(orchard_key, face1) and is_at_tree_border(orchard_key, face2);

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

    return is_at_tree_corner(orchard_key, Face::XMIN, Face::YMIN) or
           is_at_tree_corner(orchard_key, Face::XMIN, Face::YMAX) or
           is_at_tree_corner(orchard_key, Face::XMAX, Face::YMIN) or
           is_at_tree_corner(orchard_key, Face::XMAX, Face::YMAX);
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

    Kokkos::Array<int8_t, DIM> normal{ 0, 0 };

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

    return normal;

  } // get_outside_normal

  /**
   * Get orchard key of a neighbor across a corner
   * a given input octant orchard key, assuming neighbor is at same level.
   *
   * A corner is identified by (intersection of) 2 orthogonal faces
   *
   * \param[in] key is the orchard key of current octant
   * \param[in] face_x is a faceId (Face::XMIN, Face::XMAX, ....)
   * \param[in] face_y is a faceId (Face::YMIN, Face::YMAX, ....)
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
                          brick_size_t<DIM> const & brick_sizes)
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

    // face_x
    if (face_x == Face::XMIN)
    {
      xy_neigh[IX] = (xy[IX] == 0) ? last_offset(lev) : xy[IX] - length;

      // is neighbor octant in a different tree ?
      if (xy[IX] == 0)
      {
        tree_neigh_xy[IX] = tree_xy[IX] == 0 ? brick_sizes[IX] - 1 : tree_xy[IX] - 1;
      }
    }
    else if (face_x == Face::XMAX)
    {
      xy_neigh[IX] = (xy[IX] == last_offset(lev)) ? 0 : xy[IX] + length;

      // is neighbor octant in a different tree ?
      if (xy[IX] == last_offset(lev))
      {
        tree_neigh_xy[IX] = tree_xy[IX] == brick_sizes[IX] - 1 ? 0 : tree_xy[IX] + 1;
      }
    }

    // face_y
    if (face_y == Face::YMIN)
    {
      xy_neigh[IY] = (xy[IY] == 0) ? last_offset(lev) : xy[IY] - length;

      // is neighbor octant in a different tree ?
      if (xy[IY] == 0)
      {
        tree_neigh_xy[IY] = tree_xy[IY] == 0 ? brick_sizes[IY] - 1 : tree_xy[IY] - 1;
      }
    }
    else if (face_y == Face::YMAX)
    {
      xy_neigh[IY] = (xy[IY] == last_offset(lev)) ? 0 : xy[IY] + length;

      // is neighbor octant in a different tree ?
      if (xy[IY] == last_offset(lev))
      {
        tree_neigh_xy[IY] = tree_xy[IY] == brick_sizes[IY] - 1 ? 0 : tree_xy[IY] + 1;
      }
    }

    tree_neigh = encode_morton_tree(tree_neigh_xy);

    return encode_orchard(tree_neigh, xy_neigh, lev, outside_status(key));

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

    return get_corner_neighbor_key(key, face, face, brick_sizes);

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

  } // face_to_displacement

  /**
   * return a displacement vector from a face id.
   */
  KOKKOS_INLINE_FUNCTION
  static Kokkos::Array<int8_t, DIM>
  face_to_displacement(Face::face_t face)
  {

    Kokkos::Array<int8_t, DIM> displacement{ 0, 0 };

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

    Kokkos::Array<int8_t, DIM> displacement{ 0, 0 };

    face_to_displacement(displacement, face0);
    face_to_displacement(displacement, face1);

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

    printf("[%s] key: %ld | tree_morton : %d | tree_coord : %d %d | octant coord: %d %d | outside: "
           "%d%d | level: %d\n",
           str,
           key,
           morton_tree(key),
           tree_coord[IX],
           tree_coord[IY],
           oct_coord[IX],
           oct_coord[IY],
           is_touching_face_X(key),
           is_touching_face_Y(key),
           level(key));
  } // print

}; // struct orchard_key_t<2>

KALYPSSO_DISABLE_CONVERSION_WARNINGS_POP()
KALYPSSO_DISABLE_NVCC_WARNINGS_POP()
