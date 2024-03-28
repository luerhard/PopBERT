# type: ignore
# coding: utf-8
from sqlalchemy import BigInteger
from sqlalchemy import Boolean
from sqlalchemy import CheckConstraint
from sqlalchemy import Column
from sqlalchemy import DateTime
from sqlalchemy import Float
from sqlalchemy import ForeignKey
from sqlalchemy import Integer
from sqlalchemy import SmallInteger
from sqlalchemy import String
from sqlalchemy import Text
from sqlalchemy import text
from sqlalchemy import UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()
metadata = Base.metadata


class AuthGroup(Base):
    __tablename__ = "auth_group"

    id = Column(
        Integer,
        primary_key=True,
        server_default=text("nextval('auth_group_id_seq'::regclass)"),
    )
    name = Column(String(150), nullable=False, unique=True)


class AuthUser(Base):
    __tablename__ = "auth_user"

    id = Column(
        Integer,
        primary_key=True,
        server_default=text("nextval('auth_user_id_seq'::regclass)"),
    )
    password = Column(String(128), nullable=False)
    last_login = Column(DateTime(True))
    is_superuser = Column(Boolean, nullable=False)
    username = Column(String(150), nullable=False, unique=True)
    first_name = Column(String(150), nullable=False)
    last_name = Column(String(150), nullable=False)
    email = Column(String(254), nullable=False)
    is_staff = Column(Boolean, nullable=False)
    is_active = Column(Boolean, nullable=False)
    date_joined = Column(DateTime(True), nullable=False)


class DjangoCeleryResultsChordcounter(Base):
    __tablename__ = "django_celery_results_chordcounter"
    __table_args__ = (CheckConstraint("count >= 0"),)

    id = Column(
        Integer,
        primary_key=True,
        server_default=text("nextval('django_celery_results_chordcounter_id_seq'::regclass)"),
    )
    group_id = Column(String(191), nullable=False, unique=True)
    sub_tasks = Column(Text, nullable=False)
    count = Column(Integer, nullable=False)


class DjangoCeleryResultsGroupresult(Base):
    __tablename__ = "django_celery_results_groupresult"

    id = Column(
        Integer,
        primary_key=True,
        server_default=text("nextval('django_celery_results_groupresult_id_seq'::regclass)"),
    )
    group_id = Column(String(191), nullable=False, unique=True)
    date_created = Column(DateTime(True), nullable=False, index=True)
    date_done = Column(DateTime(True), nullable=False, index=True)
    content_type = Column(String(128), nullable=False)
    content_encoding = Column(String(64), nullable=False)
    result = Column(Text)


class DjangoCeleryResultsTaskresult(Base):
    __tablename__ = "django_celery_results_taskresult"

    id = Column(
        Integer,
        primary_key=True,
        server_default=text("nextval('django_celery_results_taskresult_id_seq'::regclass)"),
    )
    task_id = Column(String(191), nullable=False, unique=True)
    status = Column(String(50), nullable=False, index=True)
    content_type = Column(String(128), nullable=False)
    content_encoding = Column(String(64), nullable=False)
    result = Column(Text)
    date_done = Column(DateTime(True), nullable=False, index=True)
    traceback = Column(Text)
    meta = Column(Text)
    task_args = Column(Text)
    task_kwargs = Column(Text)
    task_name = Column(String(191), index=True)
    worker = Column(String(100), index=True)
    date_created = Column(DateTime(True), nullable=False, index=True)
    periodic_task_name = Column(String(255))


class DjangoContentType(Base):
    __tablename__ = "django_content_type"
    __table_args__ = (UniqueConstraint("app_label", "model"),)

    id = Column(
        Integer,
        primary_key=True,
        server_default=text("nextval('django_content_type_id_seq'::regclass)"),
    )
    app_label = Column(String(100), nullable=False)
    model = Column(String(100), nullable=False)


class DjangoMigration(Base):
    __tablename__ = "django_migrations"

    id = Column(
        Integer,
        primary_key=True,
        server_default=text("nextval('django_migrations_id_seq'::regclass)"),
    )
    app = Column(String(255), nullable=False)
    name = Column(String(255), nullable=False)
    applied = Column(DateTime(True), nullable=False)


class DjangoSession(Base):
    __tablename__ = "django_session"

    session_key = Column(String(40), primary_key=True, index=True)
    session_data = Column(Text, nullable=False)
    expire_date = Column(DateTime(True), nullable=False, index=True)


class RolesRole(Base):
    __tablename__ = "roles_role"

    id = Column(
        Integer,
        primary_key=True,
        server_default=text("nextval('api_role_id_seq'::regclass)"),
    )
    name = Column(String(100), nullable=False, unique=True)
    description = Column(Text, nullable=False)
    created_at = Column(DateTime(True), nullable=False)
    updated_at = Column(DateTime(True), nullable=False)


class AuthPermission(Base):
    __tablename__ = "auth_permission"
    __table_args__ = (UniqueConstraint("content_type_id", "codename"),)

    id = Column(
        Integer,
        primary_key=True,
        server_default=text("nextval('auth_permission_id_seq'::regclass)"),
    )
    name = Column(String(255), nullable=False)
    content_type_id = Column(
        ForeignKey("django_content_type.id", deferrable=True, initially="DEFERRED"),
        nullable=False,
        index=True,
    )
    codename = Column(String(100), nullable=False)

    content_type = relationship("DjangoContentType")


class AuthUserGroup(Base):
    __tablename__ = "auth_user_groups"
    __table_args__ = (UniqueConstraint("user_id", "group_id"),)

    id = Column(
        Integer,
        primary_key=True,
        server_default=text("nextval('auth_user_groups_id_seq'::regclass)"),
    )
    user_id = Column(
        ForeignKey("auth_user.id", deferrable=True, initially="DEFERRED"),
        nullable=False,
        index=True,
    )
    group_id = Column(
        ForeignKey("auth_group.id", deferrable=True, initially="DEFERRED"),
        nullable=False,
        index=True,
    )

    group = relationship("AuthGroup")
    user = relationship("AuthUser")


class AuthtokenToken(Base):
    __tablename__ = "authtoken_token"

    key = Column(String(40), primary_key=True, index=True)
    created = Column(DateTime(True), nullable=False)
    user_id = Column(
        ForeignKey("auth_user.id", deferrable=True, initially="DEFERRED"),
        nullable=False,
        unique=True,
    )

    user = relationship("AuthUser", uselist=False)


class DjangoAdminLog(Base):
    __tablename__ = "django_admin_log"
    __table_args__ = (CheckConstraint("action_flag >= 0"),)

    id = Column(
        Integer,
        primary_key=True,
        server_default=text("nextval('django_admin_log_id_seq'::regclass)"),
    )
    action_time = Column(DateTime(True), nullable=False)
    object_id = Column(Text)
    object_repr = Column(String(200), nullable=False)
    action_flag = Column(SmallInteger, nullable=False)
    change_message = Column(Text, nullable=False)
    content_type_id = Column(
        ForeignKey("django_content_type.id", deferrable=True, initially="DEFERRED"),
        index=True,
    )
    user_id = Column(
        ForeignKey("auth_user.id", deferrable=True, initially="DEFERRED"),
        nullable=False,
        index=True,
    )

    content_type = relationship("DjangoContentType")
    user = relationship("AuthUser")


class DjangoDrfFilepondStoredupload(Base):
    __tablename__ = "django_drf_filepond_storedupload"

    upload_id = Column(String(22), primary_key=True, index=True)
    file = Column(String(2048), nullable=False)
    uploaded = Column(DateTime(True), nullable=False)
    stored = Column(DateTime(True), nullable=False)
    uploaded_by_id = Column(
        ForeignKey("auth_user.id", deferrable=True, initially="DEFERRED"),
        index=True,
    )

    uploaded_by = relationship("AuthUser")


class DjangoDrfFilepondTemporaryupload(Base):
    __tablename__ = "django_drf_filepond_temporaryupload"

    upload_id = Column(String(22), primary_key=True, index=True)
    file = Column(String(100), nullable=False)
    upload_name = Column(String(512), nullable=False)
    uploaded = Column(DateTime(True), nullable=False)
    upload_type = Column(String(1), nullable=False)
    file_id = Column(String(22), nullable=False)
    uploaded_by_id = Column(
        ForeignKey("auth_user.id", deferrable=True, initially="DEFERRED"),
        index=True,
    )

    uploaded_by = relationship("AuthUser")


class DjangoDrfFilepondTemporaryuploadchunked(Base):
    __tablename__ = "django_drf_filepond_temporaryuploadchunked"

    upload_id = Column(String(22), primary_key=True, index=True)
    file_id = Column(String(22), nullable=False)
    upload_dir = Column(String(512), nullable=False)
    last_chunk = Column(Integer, nullable=False)
    total_size = Column(BigInteger, nullable=False)
    upload_name = Column(String(512), nullable=False)
    upload_complete = Column(Boolean, nullable=False)
    last_upload_time = Column(DateTime(True), nullable=False)
    uploaded_by_id = Column(
        ForeignKey("auth_user.id", deferrable=True, initially="DEFERRED"),
        index=True,
    )
    offset = Column(BigInteger, nullable=False)

    uploaded_by = relationship("AuthUser")


class ProjectsProject(Base):
    __tablename__ = "projects_project"

    id = Column(
        Integer,
        primary_key=True,
        server_default=text("nextval('api_project_id_seq'::regclass)"),
    )
    name = Column(String(100), nullable=False)
    description = Column(Text, nullable=False)
    guideline = Column(Text, nullable=False)
    created_at = Column(DateTime(True), nullable=False)
    updated_at = Column(DateTime(True), nullable=False)
    project_type = Column(String(30), nullable=False)
    random_order = Column(Boolean, nullable=False)
    collaborative_annotation = Column(Boolean, nullable=False)
    polymorphic_ctype_id = Column(
        ForeignKey("django_content_type.id", deferrable=True, initially="DEFERRED"),
        index=True,
    )
    single_class_classification = Column(Boolean, nullable=False)
    created_by_id = Column(
        ForeignKey("auth_user.id", deferrable=True, initially="DEFERRED"),
        index=True,
    )

    created_by = relationship("AuthUser")
    polymorphic_ctype = relationship("DjangoContentType")


class ProjectsBoundingboxproject(ProjectsProject):
    __tablename__ = "projects_boundingboxproject"

    project_ptr_id = Column(
        ForeignKey("projects_project.id", deferrable=True, initially="DEFERRED"),
        primary_key=True,
    )


class ProjectsImagecaptioningproject(ProjectsProject):
    __tablename__ = "projects_imagecaptioningproject"

    project_ptr_id = Column(
        ForeignKey("projects_project.id", deferrable=True, initially="DEFERRED"),
        primary_key=True,
    )


class ProjectsImageclassificationproject(ProjectsProject):
    __tablename__ = "projects_imageclassificationproject"

    project_ptr_id = Column(
        ForeignKey("projects_project.id", deferrable=True, initially="DEFERRED"),
        primary_key=True,
    )


class ProjectsIntentdetectionandslotfillingproject(ProjectsProject):
    __tablename__ = "projects_intentdetectionandslotfillingproject"

    project_ptr_id = Column(
        ForeignKey("projects_project.id", deferrable=True, initially="DEFERRED"),
        primary_key=True,
    )


class ProjectsSegmentationproject(ProjectsProject):
    __tablename__ = "projects_segmentationproject"

    project_ptr_id = Column(
        ForeignKey("projects_project.id", deferrable=True, initially="DEFERRED"),
        primary_key=True,
    )


class ProjectsSeq2seqproject(ProjectsProject):
    __tablename__ = "projects_seq2seqproject"

    project_ptr_id = Column(
        ForeignKey("projects_project.id", deferrable=True, initially="DEFERRED"),
        primary_key=True,
    )


class ProjectsSequencelabelingproject(ProjectsProject):
    __tablename__ = "projects_sequencelabelingproject"

    project_ptr_id = Column(
        ForeignKey("projects_project.id", deferrable=True, initially="DEFERRED"),
        primary_key=True,
    )
    allow_overlapping = Column(Boolean, nullable=False)
    grapheme_mode = Column(Boolean, nullable=False)
    use_relation = Column(Boolean, nullable=False)


class ProjectsSpeech2textproject(ProjectsProject):
    __tablename__ = "projects_speech2textproject"

    project_ptr_id = Column(
        ForeignKey("projects_project.id", deferrable=True, initially="DEFERRED"),
        primary_key=True,
    )


class ProjectsTextclassificationproject(ProjectsProject):
    __tablename__ = "projects_textclassificationproject"

    project_ptr_id = Column(
        ForeignKey("projects_project.id", deferrable=True, initially="DEFERRED"),
        primary_key=True,
    )


class AuthGroupPermission(Base):
    __tablename__ = "auth_group_permissions"
    __table_args__ = (UniqueConstraint("group_id", "permission_id"),)

    id = Column(
        Integer,
        primary_key=True,
        server_default=text("nextval('auth_group_permissions_id_seq'::regclass)"),
    )
    group_id = Column(
        ForeignKey("auth_group.id", deferrable=True, initially="DEFERRED"),
        nullable=False,
        index=True,
    )
    permission_id = Column(
        ForeignKey("auth_permission.id", deferrable=True, initially="DEFERRED"),
        nullable=False,
        index=True,
    )

    group = relationship("AuthGroup")
    permission = relationship("AuthPermission")


class AuthUserUserPermission(Base):
    __tablename__ = "auth_user_user_permissions"
    __table_args__ = (UniqueConstraint("user_id", "permission_id"),)

    id = Column(
        Integer,
        primary_key=True,
        server_default=text("nextval('auth_user_user_permissions_id_seq'::regclass)"),
    )
    user_id = Column(
        ForeignKey("auth_user.id", deferrable=True, initially="DEFERRED"),
        nullable=False,
        index=True,
    )
    permission_id = Column(
        ForeignKey("auth_permission.id", deferrable=True, initially="DEFERRED"),
        nullable=False,
        index=True,
    )

    permission = relationship("AuthPermission")
    user = relationship("AuthUser")


class AutoLabelingAutolabelingconfig(Base):
    __tablename__ = "auto_labeling_autolabelingconfig"

    id = Column(
        Integer,
        primary_key=True,
        server_default=text("nextval('api_autolabelingconfig_id_seq'::regclass)"),
    )
    model_name = Column(String(100), nullable=False)
    model_attrs = Column(JSONB(astext_type=Text()), nullable=False)
    template = Column(Text, nullable=False)
    label_mapping = Column(JSONB(astext_type=Text()), nullable=False)
    default = Column(Boolean, nullable=False)
    created_at = Column(DateTime(True), nullable=False)
    updated_at = Column(DateTime(True), nullable=False)
    project_id = Column(
        ForeignKey("projects_project.id", deferrable=True, initially="DEFERRED"),
        nullable=False,
        index=True,
    )
    task_type = Column(String(100), nullable=False)

    project = relationship("ProjectsProject")


class ExamplesExample(Base):
    __tablename__ = "examples_example"

    id = Column(
        Integer,
        primary_key=True,
        server_default=text("nextval('api_example_id_seq'::regclass)"),
    )
    meta = Column(JSONB(astext_type=Text()), nullable=False)
    filename = Column(String(1024), nullable=False)
    text = Column(Text)
    created_at = Column(DateTime(True), nullable=False, index=True)
    updated_at = Column(DateTime(True), nullable=False)
    annotations_approved_by_id = Column(
        ForeignKey("auth_user.id", deferrable=True, initially="DEFERRED"),
        index=True,
    )
    project_id = Column(
        ForeignKey("projects_project.id", deferrable=True, initially="DEFERRED"),
        nullable=False,
        index=True,
    )
    uuid = Column(UUID, nullable=False, unique=True)
    upload_name = Column(String(512), nullable=False)

    annotations_approved_by = relationship("AuthUser")
    project = relationship("ProjectsProject")

    labels = relationship("LabelsCategory", uselist=True)
    state = relationship("ExamplesExamplestate")


class LabelTypesCategorytype(Base):
    __tablename__ = "label_types_categorytype"
    __table_args__ = (UniqueConstraint("project_id", "text"),)

    id = Column(
        Integer,
        primary_key=True,
        server_default=text("nextval('api_categorytype_id_seq'::regclass)"),
    )
    text = Column(String(100), nullable=False, index=True)
    prefix_key = Column(String(10))
    suffix_key = Column(String(1))
    background_color = Column(String(7), nullable=False)
    text_color = Column(String(7), nullable=False)
    created_at = Column(DateTime(True), nullable=False, index=True)
    updated_at = Column(DateTime(True), nullable=False)
    project_id = Column(
        ForeignKey("projects_project.id", deferrable=True, initially="DEFERRED"),
        nullable=False,
        index=True,
    )

    project = relationship("ProjectsProject")

    def __repr__(self) -> str:
        return self.text


class LabelTypesRelationtype(Base):
    __tablename__ = "label_types_relationtype"
    __table_args__ = (UniqueConstraint("project_id", "text"),)

    id = Column(
        BigInteger,
        primary_key=True,
        server_default=text("nextval('label_types_relationtype_id_seq'::regclass)"),
    )
    text = Column(String(100), nullable=False, index=True)
    prefix_key = Column(String(10))
    suffix_key = Column(String(1))
    background_color = Column(String(7), nullable=False)
    text_color = Column(String(7), nullable=False)
    created_at = Column(DateTime(True), nullable=False, index=True)
    updated_at = Column(DateTime(True), nullable=False)
    project_id = Column(
        ForeignKey("projects_project.id", deferrable=True, initially="DEFERRED"),
        nullable=False,
        index=True,
    )

    project = relationship("ProjectsProject")


class LabelTypesSpantype(Base):
    __tablename__ = "label_types_spantype"
    __table_args__ = (UniqueConstraint("project_id", "text"),)

    id = Column(
        Integer,
        primary_key=True,
        server_default=text("nextval('api_spantype_id_seq'::regclass)"),
    )
    text = Column(String(100), nullable=False, index=True)
    prefix_key = Column(String(10))
    suffix_key = Column(String(1))
    background_color = Column(String(7), nullable=False)
    text_color = Column(String(7), nullable=False)
    created_at = Column(DateTime(True), nullable=False, index=True)
    updated_at = Column(DateTime(True), nullable=False)
    project_id = Column(
        ForeignKey("projects_project.id", deferrable=True, initially="DEFERRED"),
        nullable=False,
        index=True,
    )

    project = relationship("ProjectsProject")


class ProjectsMember(Base):
    __tablename__ = "projects_member"
    __table_args__ = (UniqueConstraint("user_id", "project_id"),)

    id = Column(
        Integer,
        primary_key=True,
        server_default=text("nextval('api_rolemapping_id_seq'::regclass)"),
    )
    created_at = Column(DateTime(True), nullable=False)
    updated_at = Column(DateTime(True), nullable=False)
    project_id = Column(
        ForeignKey("projects_project.id", deferrable=True, initially="DEFERRED"),
        nullable=False,
        index=True,
    )
    role_id = Column(
        ForeignKey("roles_role.id", deferrable=True, initially="DEFERRED"),
        nullable=False,
        index=True,
    )
    user_id = Column(
        ForeignKey("auth_user.id", deferrable=True, initially="DEFERRED"),
        nullable=False,
        index=True,
    )

    project = relationship("ProjectsProject")
    role = relationship("RolesRole")
    user = relationship("AuthUser")


class ProjectsTag(Base):
    __tablename__ = "projects_tag"

    id = Column(
        Integer,
        primary_key=True,
        server_default=text("nextval('api_tag_id_seq'::regclass)"),
    )
    text = Column(Text, nullable=False)
    project_id = Column(
        ForeignKey("projects_project.id", deferrable=True, initially="DEFERRED"),
        nullable=False,
        index=True,
    )

    project = relationship("ProjectsProject")


class ExamplesComment(Base):
    __tablename__ = "examples_comment"

    id = Column(
        Integer,
        primary_key=True,
        server_default=text("nextval('api_comment_id_seq'::regclass)"),
    )
    text = Column(Text, nullable=False)
    created_at = Column(DateTime(True), nullable=False, index=True)
    updated_at = Column(DateTime(True), nullable=False)
    user_id = Column(ForeignKey("auth_user.id", deferrable=True, initially="DEFERRED"), index=True)
    example_id = Column(
        ForeignKey("examples_example.id", deferrable=True, initially="DEFERRED"),
        nullable=False,
        index=True,
    )

    example = relationship("ExamplesExample")
    user = relationship("AuthUser")


class ExamplesExamplestate(Base):
    __tablename__ = "examples_examplestate"
    __table_args__ = (UniqueConstraint("example_id", "confirmed_by_id"),)

    id = Column(
        Integer,
        primary_key=True,
        server_default=text("nextval('api_examplestate_id_seq'::regclass)"),
    )
    confirmed_at = Column(DateTime(True), nullable=False)
    confirmed_by_id = Column(
        ForeignKey("auth_user.id", deferrable=True, initially="DEFERRED"),
        nullable=False,
        index=True,
    )
    example_id = Column(
        ForeignKey("examples_example.id", deferrable=True, initially="DEFERRED"),
        nullable=False,
        index=True,
    )

    confirmed_by = relationship("AuthUser")
    example = relationship("ExamplesExample", back_populates="state")


class LabelsBoundingbox(Base):
    __tablename__ = "labels_boundingbox"
    __table_args__ = (
        CheckConstraint("height >= (0.0)::double precision"),
        CheckConstraint("width >= (0.0)::double precision"),
        CheckConstraint("x >= (0.0)::double precision"),
        CheckConstraint("y >= (0.0)::double precision"),
    )

    id = Column(
        BigInteger,
        primary_key=True,
        server_default=text("nextval('labels_boundingbox_id_seq'::regclass)"),
    )
    uuid = Column(UUID, nullable=False, unique=True)
    prob = Column(Float(53), nullable=False)
    manual = Column(Boolean, nullable=False)
    created_at = Column(DateTime(True), nullable=False)
    updated_at = Column(DateTime(True), nullable=False)
    x = Column(Float(53), nullable=False)
    y = Column(Float(53), nullable=False)
    width = Column(Float(53), nullable=False)
    height = Column(Float(53), nullable=False)
    example_id = Column(
        ForeignKey("examples_example.id", deferrable=True, initially="DEFERRED"),
        nullable=False,
        index=True,
    )
    label_id = Column(
        ForeignKey("label_types_categorytype.id", deferrable=True, initially="DEFERRED"),
        nullable=False,
        index=True,
    )
    user_id = Column(
        ForeignKey("auth_user.id", deferrable=True, initially="DEFERRED"),
        nullable=False,
        index=True,
    )

    example = relationship("ExamplesExample")
    label = relationship("LabelTypesCategorytype")
    user = relationship("AuthUser")


class LabelsCategory(Base):
    __tablename__ = "labels_category"
    __table_args__ = (UniqueConstraint("example_id", "user_id", "label_id"),)

    id = Column(
        Integer,
        primary_key=True,
        server_default=text("nextval('api_category_id_seq'::regclass)"),
    )
    prob = Column(Float(53), nullable=False)
    manual = Column(Boolean, nullable=False)
    created_at = Column(DateTime(True), nullable=False)
    updated_at = Column(DateTime(True), nullable=False)
    example_id = Column(
        ForeignKey("examples_example.id", deferrable=True, initially="DEFERRED"),
        nullable=False,
        index=True,
    )
    label_id = Column(
        ForeignKey("label_types_categorytype.id", deferrable=True, initially="DEFERRED"),
        nullable=False,
        index=True,
    )
    user_id = Column(
        ForeignKey("auth_user.id", deferrable=True, initially="DEFERRED"),
        nullable=False,
        index=True,
    )
    uuid = Column(UUID, nullable=False, unique=True)

    example = relationship("ExamplesExample", back_populates="labels")
    label = relationship("LabelTypesCategorytype")
    user = relationship("AuthUser")

    def __repr__(self) -> str:
        return repr(self.label)


class LabelsSegmentation(Base):
    __tablename__ = "labels_segmentation"

    id = Column(
        BigInteger,
        primary_key=True,
        server_default=text("nextval('labels_segmentation_id_seq'::regclass)"),
    )
    uuid = Column(UUID, nullable=False, unique=True)
    prob = Column(Float(53), nullable=False)
    manual = Column(Boolean, nullable=False)
    created_at = Column(DateTime(True), nullable=False)
    updated_at = Column(DateTime(True), nullable=False)
    points = Column(JSONB(astext_type=Text()), nullable=False)
    example_id = Column(
        ForeignKey("examples_example.id", deferrable=True, initially="DEFERRED"),
        nullable=False,
        index=True,
    )
    label_id = Column(
        ForeignKey("label_types_categorytype.id", deferrable=True, initially="DEFERRED"),
        nullable=False,
        index=True,
    )
    user_id = Column(
        ForeignKey("auth_user.id", deferrable=True, initially="DEFERRED"),
        nullable=False,
        index=True,
    )

    example = relationship("ExamplesExample")
    label = relationship("LabelTypesCategorytype")
    user = relationship("AuthUser")


class LabelsSpan(Base):
    __tablename__ = "labels_span"
    __table_args__ = (
        CheckConstraint("end_offset >= 0"),
        CheckConstraint("start_offset < end_offset"),
        CheckConstraint("start_offset >= 0"),
    )

    id = Column(
        Integer,
        primary_key=True,
        server_default=text("nextval('api_span_id_seq'::regclass)"),
    )
    prob = Column(Float(53), nullable=False)
    manual = Column(Boolean, nullable=False)
    created_at = Column(DateTime(True), nullable=False)
    updated_at = Column(DateTime(True), nullable=False)
    start_offset = Column(Integer, nullable=False)
    end_offset = Column(Integer, nullable=False)
    example_id = Column(
        ForeignKey("examples_example.id", deferrable=True, initially="DEFERRED"),
        nullable=False,
        index=True,
    )
    label_id = Column(
        ForeignKey("label_types_spantype.id", deferrable=True, initially="DEFERRED"),
        nullable=False,
        index=True,
    )
    user_id = Column(
        ForeignKey("auth_user.id", deferrable=True, initially="DEFERRED"),
        nullable=False,
        index=True,
    )
    uuid = Column(UUID, nullable=False, unique=True)

    example = relationship("ExamplesExample")
    label = relationship("LabelTypesSpantype")
    user = relationship("AuthUser")


class LabelsTextlabel(Base):
    __tablename__ = "labels_textlabel"
    __table_args__ = (UniqueConstraint("example_id", "user_id", "text"),)

    id = Column(
        Integer,
        primary_key=True,
        server_default=text("nextval('api_textlabel_id_seq'::regclass)"),
    )
    prob = Column(Float(53), nullable=False)
    manual = Column(Boolean, nullable=False)
    created_at = Column(DateTime(True), nullable=False)
    updated_at = Column(DateTime(True), nullable=False)
    text = Column(Text, nullable=False)
    example_id = Column(
        ForeignKey("examples_example.id", deferrable=True, initially="DEFERRED"),
        nullable=False,
        index=True,
    )
    user_id = Column(
        ForeignKey("auth_user.id", deferrable=True, initially="DEFERRED"),
        nullable=False,
        index=True,
    )
    uuid = Column(UUID, nullable=False, unique=True)

    example = relationship("ExamplesExample")
    user = relationship("AuthUser")


class LabelsRelation(Base):
    __tablename__ = "labels_relation"

    id = Column(
        BigInteger,
        primary_key=True,
        server_default=text("nextval('labels_relationnew_id_seq'::regclass)"),
    )
    prob = Column(Float(53), nullable=False)
    manual = Column(Boolean, nullable=False)
    created_at = Column(DateTime(True), nullable=False)
    updated_at = Column(DateTime(True), nullable=False)
    example_id = Column(
        ForeignKey("examples_example.id", deferrable=True, initially="DEFERRED"),
        nullable=False,
        index=True,
    )
    from_id_id = Column(
        ForeignKey("labels_span.id", deferrable=True, initially="DEFERRED"),
        nullable=False,
        index=True,
    )
    to_id_id = Column(
        ForeignKey("labels_span.id", deferrable=True, initially="DEFERRED"),
        nullable=False,
        index=True,
    )
    type_id = Column(
        ForeignKey("label_types_relationtype.id", deferrable=True, initially="DEFERRED"),
        nullable=False,
        index=True,
    )
    user_id = Column(
        ForeignKey("auth_user.id", deferrable=True, initially="DEFERRED"),
        nullable=False,
        index=True,
    )
    uuid = Column(UUID, nullable=False, unique=True)

    example = relationship("ExamplesExample")
    from_id = relationship("LabelsSpan", primaryjoin="LabelsRelation.from_id_id == LabelsSpan.id")
    to_id = relationship("LabelsSpan", primaryjoin="LabelsRelation.to_id_id == LabelsSpan.id")
    type = relationship("LabelTypesRelationtype")
    user = relationship("AuthUser")
