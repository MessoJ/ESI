from alembic import op
import sqlalchemy as sa

revision = '0002_provenance'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        'series_points',
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('date', sa.Date, index=True),
        sa.Column('series', sa.String(64), index=True),
        sa.Column('country', sa.String(8), index=True),
        sa.Column('value', sa.Float),
    )
    op.create_index('sp_idx', 'series_points', ['series','country','date'])
    op.create_table(
        'data_provenance',
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('run_id', sa.String(64), index=True),
        sa.Column('source_url', sa.Text),
        sa.Column('created_at', sa.DateTime),
        sa.Column('notes', sa.Text),
    )


def downgrade():
    op.drop_table('data_provenance')
    op.drop_index('sp_idx', table_name='series_points')
    op.drop_table('series_points')



