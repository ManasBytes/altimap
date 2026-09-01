import { useRef, useState } from 'react';

export default function DropZone({ onUpload, statusText, error, busy }) {
  const inputRef = useRef(null);
  const [over, setOver] = useState(false);

  const handleFiles = (files) => {
    if (files?.[0]) onUpload(files[0]);
  };

  return (
    <>
      <div
        id="drop"
        className={[over && 'over', busy && 'busy'].filter(Boolean).join(' ')}
        onClick={() => inputRef.current?.click()}
        onDragEnter={(e) => { e.preventDefault(); setOver(true); }}
        onDragOver={(e) => { e.preventDefault(); setOver(true); }}
        onDragLeave={(e) => { e.preventDefault(); setOver(false); }}
        onDrop={(e) => { e.preventDefault(); setOver(false); handleFiles(e.dataTransfer?.files); }}
      >
        <p className="dz-main">Drop imagery here</p>
        <p className="dz-sub">or click to choose a file</p>
        <input
          ref={inputRef}
          type="file"
          accept=".tif,.tiff,.png,.jpg,.jpeg"
          hidden
          onChange={(e) => { handleFiles(e.target.files); e.target.value = ''; }}
        />
      </div>
      <p className={`hint${error ? ' alert' : ''}`}>
        {statusText || 'Georeferenced input also gets a CRS readout and an elevation-calibration attempt.'}
      </p>
    </>
  );
}
