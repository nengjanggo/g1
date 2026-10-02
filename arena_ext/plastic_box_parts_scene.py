'''
"Move parts from the plastic box to the shelf." (G1_WBT_*_Plastic_Box_Parts) 장면용 asset.

- 선반(노란 판 + 검은 프레임), 수납함(반투명 회색)은 직육면체로, 방은 회색 목재 텍스처 바닥과 둘러싼 짙은 회색 벽으로 만든 독립 USD다.
  Arena와 무관한 일반 USD라 다른 sim에서도 그대로 쓸 수 있다.
- 같은 파일에서 Arena asset으로 등록한다 (arena_ext/g1_dex1.py가 import하면 등록됨).
- 부품은 실제 asset을 찾기 전까지 쓰는 위가 파인 검은 쟁반 모양 USD다. 골판지 상자는 Arena procedural cuboid 방식으로 만든다.

USD 생성 (Arena venv, sim 없이 실행):
    third_party/IsaacLab-Arena/.venv/bin/python arena_ext/plastic_box_parts_scene.py
'''

from pathlib import Path

GENERATED_DIR: Path = Path(__file__).resolve().parents[1] / 'assets/generated/plastic_box_parts'

# 치수 [m]. 데이터셋 머리 카메라 영상으로 눈대중한 값이라 장면 검토 후 조정한다
# 선반: x 방향으로 긴 조립식 선반. 단 높이는 선반 판 윗면 기준
SHELF_LENGTH: float = 2.0
SHELF_DEPTH: float = 0.45
# 실제 선반은 3단이고 맨 윗단이 G1 키(약 1.32m)보다 5~10cm 높다
SHELF_HEIGHT: float = 1.42
SHELF_LEVEL_HEIGHTS: list[float] = [0.15, 0.78, 1.4]
SHELF_NUM_BAYS: int = 2
SHELF_BOARD_THICKNESS: float = 0.03
SHELF_POST_WIDTH: float = 0.025

# 수납함: 위가 열린 사각 상자 (바깥 치수)
TOTE_SIZE: tuple[float, float, float] = (0.6, 0.42, 0.32)
TOTE_WALL_THICKNESS: float = 0.012

# 골판지 상자: 수납함 안에 들어가고 수납함 위로 튀어나오는 크기
CARDBOARD_SIZE: tuple[float, float, float] = (0.54, 0.36, 0.4)

# 부품: 실제 부품 asset을 찾기 전까지 쓰는, 위가 파인 쟁반 모양 (바깥 치수)
PART_SIZE: tuple[float, float, float] = (0.32, 0.11, 0.045)
PART_WALL_THICKNESS: float = 0.01
PART_MASS: float = 0.1

# 바닥: 정사각 평면, 목재 텍스처를 FLOOR_TEXTURE_TILE m마다 반복
FLOOR_LENGTH: float = 10.0
FLOOR_TEXTURE_TILE: float = 1.0

# 바닥 텍스처 원본 (Poly Haven laminate_floor_02, CC0. README의 다운로드 명령 참고)을 흑백으로 바꾼 뒤 밝기를 곱한다
FLOOR_TEXTURE_SOURCE: Path = Path(__file__).resolve().parents[1] / 'assets/textures/laminate_floor_02_diff_1k.jpg'
FLOOR_TEXTURE_BRIGHTNESS: float = 0.85

# 벽: 바닥 가장자리를 따라 방 전체를 둘러싸는 4개의 판 (바깥면이 바닥 끝에 맞음)
WALL_THICKNESS: float = 0.08
WALL_HEIGHT: float = 2.6

# 천장 조명: 아래를 향하는 사각형 조명 (방 좌표 xy, 높이는 벽 높이 바로 아래). 노란 선반 판의 광택이 보이게 한다
CEILING_LIGHT_SIZE: tuple[float, float] = (1.2, 0.3)
CEILING_LIGHT_INTENSITY: float = 15000.0
CEILING_LIGHT_XY: list[tuple[float, float]] = [(0.5, 2.8), (2.0, 2.8), (0.5, 4.2), (2.0, 4.2)]

# 색 (linear RGB)
SHELF_BOARD_COLOR: tuple[float, float, float] = (1.0, 0.55, 0.0)
# 노란 판은 광택 있는 도장 철판이라 거칠기를 낮춤 (다른 재질은 0.6)
SHELF_BOARD_ROUGHNESS: float = 0.2
SHELF_FRAME_COLOR: tuple[float, float, float] = (0.05, 0.05, 0.05)
TOTE_COLOR: tuple[float, float, float] = (0.85, 0.87, 0.9)
TOTE_OPACITY: float = 0.6
CARDBOARD_COLOR: tuple[float, float, float] = (0.6, 0.3, 0.09)
PART_COLOR: tuple[float, float, float] = (0.03, 0.03, 0.03)
WALL_COLOR: tuple[float, float, float] = (0.15, 0.15, 0.16)

# 장면 조명 (Arena 기본 dome light는 color 0.75, intensity 1500)
DOME_LIGHT_COLOR: tuple[float, float, float] = (0.9, 0.9, 0.9)
DOME_LIGHT_INTENSITY: float = 2500.0


def add_box(
    stage,
    path: str,
    size: tuple[float, float, float],
    center: tuple[float, float, float],
    material,
) -> None:
    '''stage의 path에 크기 size, 중심 center인 직육면체를 만들고 material과 collision을 붙인다.'''
    from pxr import Gf, UsdGeom, UsdPhysics, UsdShade

    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    cube.AddTranslateOp().Set(Gf.Vec3d(*center))
    cube.AddScaleOp().Set(Gf.Vec3f(*size))
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(material)


def add_material(
    stage,
    path: str,
    color: tuple[float, float, float],
    opacity: float = 1.0,
    roughness: float = 0.6,
):
    '''stage의 path에 diffuse color, opacity, roughness를 가진 UsdPreviewSurface material을 만들어 반환한다.'''
    from pxr import Sdf, UsdShade

    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f'{path}/Shader')
    shader.CreateIdAttr('UsdPreviewSurface')
    shader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set(color)
    shader.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(roughness)
    shader.CreateInput('opacity', Sdf.ValueTypeNames.Float).Set(opacity)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), 'surface')
    return material


def new_stage(
    usd_path: Path,
):
    '''Z-up, meter 단위의 새 stage를 만들고 root Xform '/Root'를 default prim으로 정해 반환한다.'''
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.CreateNew(str(usd_path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, '/Root').GetPrim())
    return stage


def build_shelf_usd(
    usd_path: Path,
) -> None:
    '''선반 USD를 만든다. 원점은 선반 바닥면 중심, 길이 방향 x, 앞면은 -y 쪽.'''
    stage = new_stage(usd_path)
    board_material = add_material(stage, '/Root/Looks/Board', SHELF_BOARD_COLOR, roughness=SHELF_BOARD_ROUGHNESS)
    frame_material = add_material(stage, '/Root/Looks/Frame', SHELF_FRAME_COLOR)
    # 기둥: bay 경계마다 앞/뒤 한 쌍
    for i in range(SHELF_NUM_BAYS + 1):
        x: float = -SHELF_LENGTH / 2 + i * SHELF_LENGTH / SHELF_NUM_BAYS
        x = min(max(x, -SHELF_LENGTH / 2 + SHELF_POST_WIDTH / 2), SHELF_LENGTH / 2 - SHELF_POST_WIDTH / 2)
        for j, y in enumerate([-SHELF_DEPTH / 2 + SHELF_POST_WIDTH / 2, SHELF_DEPTH / 2 - SHELF_POST_WIDTH / 2]):
            add_box(stage, f'/Root/post_{i}_{j}', (SHELF_POST_WIDTH, SHELF_POST_WIDTH, SHELF_HEIGHT), (x, y, SHELF_HEIGHT / 2), frame_material)
    # 선반 판: 단마다 전체 길이 하나
    for k, z in enumerate(SHELF_LEVEL_HEIGHTS):
        add_box(stage, f'/Root/board_{k}', (SHELF_LENGTH, SHELF_DEPTH, SHELF_BOARD_THICKNESS), (0.0, 0.0, z - SHELF_BOARD_THICKNESS / 2), board_material)
    stage.Save()


def build_tote_usd(
    usd_path: Path,
) -> None:
    '''위가 열린 반투명 수납함 USD를 만든다. 원점은 수납함 바닥면 중심.'''
    stage = new_stage(usd_path)
    material = add_material(stage, '/Root/Looks/Tote', TOTE_COLOR, TOTE_OPACITY)
    length, width, height = TOTE_SIZE
    t: float = TOTE_WALL_THICKNESS
    add_box(stage, '/Root/bottom', (length, width, t), (0.0, 0.0, t / 2), material)
    add_box(stage, '/Root/wall_front', (length, t, height), (0.0, -width / 2 + t / 2, height / 2), material)
    add_box(stage, '/Root/wall_back', (length, t, height), (0.0, width / 2 - t / 2, height / 2), material)
    add_box(stage, '/Root/wall_left', (t, width, height), (-length / 2 + t / 2, 0.0, height / 2), material)
    add_box(stage, '/Root/wall_right', (t, width, height), (length / 2 - t / 2, 0.0, height / 2), material)
    stage.Save()


def build_floor_usd(
    usd_path: Path,
) -> None:
    '''회색 목재 텍스처를 입힌 바닥 평면, 가장자리를 둘러싸는 벽 4개, 천장 조명으로 된 방 USD를 만든다. 바닥 윗면이 z=0이고 원점이 중심.'''
    from PIL import Image, ImageOps
    from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdPhysics, UsdShade, Vt

    # 원본 텍스처를 흑백으로 바꾸고 밝기를 낮춰 USD 옆에 저장
    texture_path: Path = usd_path.with_name('floor_grey_wood.png')
    grey: Image.Image = ImageOps.grayscale(Image.open(FLOOR_TEXTURE_SOURCE)).point(lambda v: int(v * FLOOR_TEXTURE_BRIGHTNESS))
    grey.convert('RGB').save(texture_path)

    stage = new_stage(usd_path)
    # UV 좌표가 있는 정사각 평면 mesh (UV를 타일 수만큼 늘려 텍스처를 반복)
    half: float = FLOOR_LENGTH / 2
    num_tiles: float = FLOOR_LENGTH / FLOOR_TEXTURE_TILE
    mesh = UsdGeom.Mesh.Define(stage, '/Root/floor')
    mesh.CreatePointsAttr([(-half, -half, 0.0), (half, -half, 0.0), (half, half, 0.0), (-half, half, 0.0)])
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    mesh.CreateNormalsAttr([(0.0, 0.0, 1.0)] * 4)
    st = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar('st', Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
    st.Set(Vt.Vec2fArray([Gf.Vec2f(0, 0), Gf.Vec2f(num_tiles, 0), Gf.Vec2f(num_tiles, num_tiles), Gf.Vec2f(0, num_tiles)]))
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())

    # st primvar로 텍스처를 읽어 diffuseColor에 연결한 material
    material = UsdShade.Material.Define(stage, '/Root/Looks/Floor')
    shader = UsdShade.Shader.Define(stage, '/Root/Looks/Floor/Shader')
    shader.CreateIdAttr('UsdPreviewSurface')
    shader.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(0.7)
    st_reader = UsdShade.Shader.Define(stage, '/Root/Looks/Floor/STReader')
    st_reader.CreateIdAttr('UsdPrimvarReader_float2')
    st_reader.CreateInput('varname', Sdf.ValueTypeNames.Token).Set('st')
    texture = UsdShade.Shader.Define(stage, '/Root/Looks/Floor/Texture')
    texture.CreateIdAttr('UsdUVTexture')
    texture.CreateInput('file', Sdf.ValueTypeNames.Asset).Set(f'./{texture_path.name}')
    texture.CreateInput('wrapS', Sdf.ValueTypeNames.Token).Set('repeat')
    texture.CreateInput('wrapT', Sdf.ValueTypeNames.Token).Set('repeat')
    texture.CreateInput('st', Sdf.ValueTypeNames.Float2).ConnectToSource(st_reader.ConnectableAPI(), 'result')
    texture.CreateOutput('rgb', Sdf.ValueTypeNames.Float3)
    shader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).ConnectToSource(texture.ConnectableAPI(), 'rgb')
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), 'surface')
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)

    # 바닥 가장자리를 따라 벽 4개
    wall_material = add_material(stage, '/Root/Looks/Wall', WALL_COLOR)
    wall_center: float = half - WALL_THICKNESS / 2
    for name, size, center in [
        ('wall_px', (WALL_THICKNESS, FLOOR_LENGTH, WALL_HEIGHT), (wall_center, 0.0, WALL_HEIGHT / 2)),
        ('wall_nx', (WALL_THICKNESS, FLOOR_LENGTH, WALL_HEIGHT), (-wall_center, 0.0, WALL_HEIGHT / 2)),
        ('wall_py', (FLOOR_LENGTH, WALL_THICKNESS, WALL_HEIGHT), (0.0, wall_center, WALL_HEIGHT / 2)),
        ('wall_ny', (FLOOR_LENGTH, WALL_THICKNESS, WALL_HEIGHT), (0.0, -wall_center, WALL_HEIGHT / 2)),
    ]:
        add_box(stage, f'/Root/{name}', size, center, wall_material)

    # 천장 조명 (RectLight는 local -z 방향으로 빛을 내므로 회전 없이 아래를 향함)
    for i, (x, y) in enumerate(CEILING_LIGHT_XY):
        light = UsdLux.RectLight.Define(stage, f'/Root/ceiling_light_{i}')
        light.CreateWidthAttr(CEILING_LIGHT_SIZE[0])
        light.CreateHeightAttr(CEILING_LIGHT_SIZE[1])
        light.CreateIntensityAttr(CEILING_LIGHT_INTENSITY)
        light.AddTranslateOp().Set(Gf.Vec3d(x, y, WALL_HEIGHT - 0.05))
    stage.Save()


def build_part_usd(
    usd_path: Path,
) -> None:
    '''위가 파인 쟁반 모양의 검은 부품 USD를 만든다. root가 하나의 rigid body이고 원점은 부품 중심.'''
    from pxr import UsdPhysics

    stage = new_stage(usd_path)
    root = stage.GetDefaultPrim()
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(PART_MASS)
    material = add_material(stage, '/Root/Looks/Part', PART_COLOR)
    length, width, height = PART_SIZE
    t: float = PART_WALL_THICKNESS
    add_box(stage, '/Root/bottom', (length, width, t), (0.0, 0.0, -height / 2 + t / 2), material)
    add_box(stage, '/Root/wall_front', (length, t, height), (0.0, -width / 2 + t / 2, 0.0), material)
    add_box(stage, '/Root/wall_back', (length, t, height), (0.0, width / 2 - t / 2, 0.0), material)
    add_box(stage, '/Root/wall_left', (t, width, height), (-length / 2 + t / 2, 0.0, 0.0), material)
    add_box(stage, '/Root/wall_right', (t, width, height), (length / 2 - t / 2, 0.0, 0.0), material)
    stage.Save()


if __name__ == '__main__':
    # 선반, 수납함, 방(바닥 + 벽), 부품 USD를 생성
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    for build, name in [(build_shelf_usd, 'shelf.usda'), (build_tote_usd, 'tote.usda'), (build_floor_usd, 'room.usda'), (build_part_usd, 'part.usda')]:
        (GENERATED_DIR / name).unlink(missing_ok=True)
        build(GENERATED_DIR / name)
else:
    # Arena asset 등록 (sim app이 뜬 뒤 import될 때만)
    import isaaclab.sim as sim_utils
    from isaaclab.assets import RigidObjectCfg

    from isaaclab_arena.assets.background_library import LibraryBackground
    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_library import DomeLight, LibraryObject
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.assets.register import register_asset
    from isaaclab_arena.utils.pose import Pose

    @register_asset
    class PlasticBoxPartsRoom(LibraryBackground):
        '''장면 방: 바닥과 둘러싼 벽 (생성된 room.usda).'''

        name = 'plastic_box_parts_room'
        tags = ['background']
        usd_path = str(GENERATED_DIR / 'room.usda')
        object_min_z = 0.0

    @register_asset
    class BoltlessShelf(LibraryObject):
        '''고정된 조립식 선반 (생성된 shelf.usda).'''

        name = 'boltless_shelf'
        tags = ['object', 'fixture']
        usd_path = str(GENERATED_DIR / 'shelf.usda')
        object_type = ObjectType.BASE

    @register_asset
    class StorageTote(LibraryObject):
        '''고정된 반투명 수납함 (생성된 tote.usda).'''

        name = 'storage_tote'
        tags = ['object', 'fixture']
        usd_path = str(GENERATED_DIR / 'tote.usda')
        object_type = ObjectType.BASE

    class ProceduralCuboidObject(Object):
        '''class 속성 spawn_cfg(CuboidCfg)로 만드는 rigid 직육면체 물체.'''

        spawn_cfg: sim_utils.CuboidCfg

        def __init__(
            self,
            instance_name: str | None = None,
            prim_path: str | None = None,
            initial_pose: Pose | None = None,
        ):
            '''instance_name을 이름과 prim 이름으로 쓰는 rigid 물체를 만든다.'''
            resolved_name: str = instance_name if instance_name is not None else self.name
            super().__init__(
                name=resolved_name,
                prim_path=prim_path if prim_path is not None else f'{{ENV_REGEX_NS}}/{resolved_name}',
                object_type=ObjectType.RIGID,
                usd_path='',
                initial_pose=initial_pose,
            )

        def _generate_rigid_cfg(
            self,
        ) -> RigidObjectCfg:
            '''spawn_cfg로 RigidObjectCfg를 만들어 초기 pose를 붙여 반환한다.'''
            return self._add_initial_pose_to_cfg(RigidObjectCfg(prim_path=self.prim_path, spawn=self.spawn_cfg, **self.asset_cfg_addon))

    @register_asset
    class CardboardInsert(ProceduralCuboidObject):
        '''수납함 안에 고정된 골판지 상자 (kinematic).'''

        name = 'cardboard_insert'
        tags = ['object', 'procedural']
        spawn_cfg = sim_utils.CuboidCfg(
            size=CARDBOARD_SIZE,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=CARDBOARD_COLOR),
        )

    @register_asset
    class PartTray(LibraryObject):
        '''위가 파인 쟁반 모양의 검은 부품 (생성된 part.usda, rigid, 0.1 kg).'''

        name = 'part_tray'
        tags = ['object']
        usd_path = str(GENERATED_DIR / 'part.usda')
        object_type = ObjectType.RIGID

    @register_asset
    class BrightDomeLight(DomeLight):
        '''Arena 기본 dome light보다 밝은 장면 조명.'''

        name = 'bright_dome_light'

        def __init__(
            self,
            instance_name: str | None = None,
            prim_path: str | None = DomeLight.default_prim_path,
            initial_pose: Pose | None = None,
        ):
            '''DOME_LIGHT_COLOR, DOME_LIGHT_INTENSITY로 dome light를 만든다.'''
            super().__init__(
                instance_name=instance_name,
                prim_path=prim_path,
                initial_pose=initial_pose,
                spawner_cfg=sim_utils.DomeLightCfg(color=DOME_LIGHT_COLOR, intensity=DOME_LIGHT_INTENSITY),
            )
