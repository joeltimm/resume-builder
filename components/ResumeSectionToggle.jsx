import React from 'react';
import PropTypes from 'prop-types';

function ResumeSectionToggle({ sectionName, isEnabledByDefault = false }) {
  return (
    <div style={{ margin: '10px', padding: '8px', border: '1px solid #ccc', borderRadius: '4px' }}>
      <label style={{ display: 'flex', alignItems: 'center', cursor: 'pointer' }}>
        <input
          type="checkbox"
          name={sectionName.toLowerCase().replace(/\s+/g, '-')}
          defaultChecked={isEnabledByDefault}
        />
        <span style={{ marginLeft: '8px', fontWeight: 'bold' }}>{sectionName}</span>
      </label>
    </div>
  );
}

ResumeSectionToggle.propTypes = {
  sectionName: PropTypes.string.isRequired,
  isEnabledByDefault: PropTypes.bool,
};

export default ResumeSectionToggle;